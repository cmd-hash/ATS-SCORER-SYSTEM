import asyncio
import logging
from typing import List, Optional
from functools import partial

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from api.auth import get_current_user
from models.schemas import AnalysisResponse, ComponentScores, JDComparison, SkillValidationDetails
from utils.file_utils import (
    get_default_grammar_results,
    get_default_location_results,
    get_default_skill_validation_results,
)

logger = logging.getLogger('ats_resume_scorer')

router = APIRouter(prefix='/api/v1', tags=['Analysis'])


def _clean(text: str) -> str:
    for prefix in ('✅', '🌟', '❌', '⚠️', '📝', '🔴', '🟡', '🟢', '🟠', '👍'):
        text = text.lstrip(prefix)
    return text.strip()


@router.post('/analyze-resume', response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(..., description='Resume file — PDF or DOCX, max 5 MB'),
    job_description: str = Form('', description='Job description text (optional)'),
    user_id: str = Depends(get_current_user),
):
    warnings: List[str] = []

    nlp      = request.app.state.nlp
    embedder = request.app.state.embedder

    # ── Phase 1: Parse the uploaded file ──────────────────────────────────────
    try:
        file_bytes = await resume.read()
        filename   = resume.filename or 'resume'

        from services.resume_parser import (
            FileParsingError,
            FileValidationError,
            parse_resume_file,
        )

        # parse_resume_file is sync — run in thread pool so we don't block
        loop = asyncio.get_event_loop()
        resume_text, _metadata = await loop.run_in_executor(
            None, parse_resume_file, file_bytes, filename
        )
        logger.info(f"Parsed '{filename}': {len(resume_text)} chars extracted")

    except Exception as exc:
        logger.error(f'File parsing failed: {exc}')
        raise HTTPException(
            status_code=422,
            detail=f'Could not read or parse the resume: {exc}',
        )

    # ── Phase 2: Full Analysis Pipeline ───────────────────────────────────────
    try:
        from services.resume_analyzer import analyze_full_resume

        # analyze_full_resume is CPU/IO bound sync — MUST run in executor
        # to avoid blocking the async event loop and causing timeouts
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                analyze_full_resume,
                resume_text=resume_text,
                nlp=nlp,
                embedder=embedder,
                job_description=job_description,
            )
        )

    except Exception as exc:
        logger.error(f'Full analysis pipeline failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Analysis pipeline failed: {exc}')

    # ── Phase 3: Build response ────────────────────────────────────────────────
    from models.schemas import ComponentScores

    jd_comparison_result = None
    if result.get('jd_comparison'):
        jd_comparison_result = JDComparison(
            match_percentage   =round(float(result['jd_comparison'].get('match_percentage',    0.0)), 1),
            semantic_similarity=round(float(result['jd_comparison'].get('semantic_similarity', 0.0)), 3),
            matched_keywords   =result['jd_comparison'].get('matched_keywords', [])[:20],
            missing_keywords   =result['jd_comparison'].get('missing_keywords', [])[:15],
            skills_gap         =result['jd_comparison'].get('skills_gap',       [])[:10],
        )

    detailed_fb = result.get('detailed_feedback', [])

    svd_raw = result.get('skill_validation_details') or {}
    skill_val_details = SkillValidationDetails(
        validated       =svd_raw.get('validated',       []),
        unvalidated     =svd_raw.get('unvalidated',     []),
        total           =svd_raw.get('total',           0),
        validated_count =svd_raw.get('validated_count', 0),
        validation_pct  =svd_raw.get('validation_pct',  0.0),
    )

    response = AnalysisResponse(
        ATS_score         =result['ats_score'],
        component_scores  =ComponentScores(**result['component_scores']),
        issues_summary    =result['issues_summary'],
        detailed_feedback =detailed_fb,
        jd_match_analysis =jd_comparison_result,
        skill_validation_details=skill_val_details,

        # Retro-compatibility fields
        ats_score        =result['ats_score'],
        keyword_match    =jd_comparison_result.match_percentage if jd_comparison_result else 0.0,
        missing_keywords =result.get('missing_keywords', []),
        matched_keywords =result.get('matched_keywords', []),
        skills           =list(result.get('skills', [])[:20]),
        jd_comparison    =jd_comparison_result,
        interpretation   =result.get('interpretation', ''),
    )

    # ── Phase 4: Save to history (non-blocking) ────────────────────────────────
    try:
        from database.supabase_db import save_analysis
        await save_analysis(user_id, filename, result)
    except Exception as exc:
        logger.warning(f'History save failed (non-blocking): {exc}')

    return response


@router.get('/health')
async def health_check(request: Request):
    return {
        'status':          'healthy',
        'nlp_loaded':      request.app.state.nlp      is not None,
        'embedder_loaded': request.app.state.embedder is not None,
    }


@router.get('/history')
async def get_history(user_id: str = Depends(get_current_user)):
    from database.supabase_db import get_user_history
    try:
        return await get_user_history(user_id)
    except Exception as exc:
        logger.error(f'History fetch failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Could not load history: {exc}')


@router.delete('/history/{analysis_id}')
async def delete_history_entry(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    from database.supabase_db import delete_analysis
    try:
        success = await delete_analysis(analysis_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail='Analysis not found or not owned by this user.')
        return {'status': 'deleted', 'id': analysis_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f'History delete failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Could not delete: {exc}')


@router.post('/generate-pdf')
async def generate_pdf(
    data: AnalysisResponse,
    user_id: str = Depends(get_current_user),
):
    from services.report_generator import generate_html_reports
    from services.pdf_export import generate_combined_pdf
    from fastapi.responses import Response

    try:
        loop = asyncio.get_event_loop()
        html_docs = await loop.run_in_executor(
            None, generate_html_reports, data.model_dump()
        )
        pdf_bytes = await loop.run_in_executor(
            None, generate_combined_pdf, html_docs
        )
        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={'Content-Disposition': 'attachment; filename=ats_report.pdf'},
        )
    except Exception as e:
        logger.error(f'Failed to generate PDF: {e}')
        raise HTTPException(status_code=500, detail=f'Failed to generate PDF: {e}')


@router.get('/history/{analysis_id}/pdf')
async def generate_history_pdf(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    from database.supabase_db import get_user_history
    from services.report_generator import generate_html_reports
    from services.pdf_export import generate_combined_pdf
    from fastapi.responses import Response

    history = await get_user_history(user_id)
    analysis_data = next(
        (item['analysis_result'] for item in history if item['id'] == analysis_id),
        None,
    )

    if not analysis_data:
        raise HTTPException(status_code=404, detail='Analysis not found')

    try:
        loop = asyncio.get_event_loop()
        html_docs = await loop.run_in_executor(
            None, generate_html_reports, analysis_data
        )
        pdf_bytes = await loop.run_in_executor(
            None, generate_combined_pdf, html_docs
        )
        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename=ats_report_{analysis_id}.pdf'},
        )
    except Exception as e:
        logger.error(f'Failed to generate PDF for history: {e}')
        raise HTTPException(status_code=500, detail=f'Failed to generate PDF: {e}')
