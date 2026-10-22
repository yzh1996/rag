"""只对没有上下文、缺少指代对象的短问题请求补充信息。"""
import re

from config.settings import settings


def clarification_for(question, history):
    if not settings.CLARIFICATION_ENABLED or history:
        return None
    text = re.sub(r"[\s，。！？?!.]", "", question)
    if re.fullmatch(r"(?:(?:那个|这个|它|那|这)(?:呢|是什么|怎么办|怎么弄|怎么处理|怎么做|如何处理|可以吗|怎么申请)?|怎么弄|怎么办|如何处理|帮我处理|继续|然后呢|为什么)", text):
        return "请补充你指的是哪项业务、哪份文档或哪个具体问题，我再帮你查找处理方法。"
    return None
