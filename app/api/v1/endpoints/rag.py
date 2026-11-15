from fastapi import APIRouter, HTTPException, status
from app.rag.qa_engine import get_rag_service
from app.schemas.rag import RAGAnswerResponse, RAGQuestionRequest

router = APIRouter(prefix="/rag", tags=["RAG Question Answering"])


@router.post(
    "/ask",
    response_model=RAGAnswerResponse,
    summary="Ask natural language questions about tender specifications with verified page citations",
    status_code=status.HTTP_200_OK,
)
def ask_tender_question(request: RAGQuestionRequest):
    """
    RAG (Retrieval-Augmented Generation) endpoint:
    Retrieves semantic chunks & table rows from Qdrant, synthesizes a grounded answer,
    and returns exact verified page citations for every point.
    """
    try:
        rag_service = get_rag_service()
        response = rag_service.ask(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء معالجة السؤال الذكي: {str(e)}",
        )
