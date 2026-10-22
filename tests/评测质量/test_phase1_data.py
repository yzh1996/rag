"""第一阶段业务数据和反馈接口的无外部服务测试。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base
from core.db_models import EvalCase, QAInteraction
from core.phase1_services import create_eval_case, get_interaction, save_feedback


def test_feedback_and_eval_case_persist(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'phase1-test.db'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = session_factory()
    try:
        interaction = QAInteraction(
            session_id="test-session",
            question="测试问题",
            answer="测试答案",
            sources=[],
        )
        db.add(interaction)
        db.flush()
        interaction_id = interaction.id
        feedback = save_feedback(db, interaction_id, "negative", "incorrect", "需要修正")
        case = create_eval_case(
            db,
            question="年假如何计算？",
            expected_answer="按制度计算",
            expected_sources=["leave-policy"],
            tags=["制度"],
        )
        assert get_interaction(db, interaction_id) is not None
        assert db.get(type(feedback), feedback.id).rating == "negative"
        assert db.get(EvalCase, case.id).question == "年假如何计算？"
    finally:
        db.close()
