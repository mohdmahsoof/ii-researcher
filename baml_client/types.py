###############################################################################
#
#  Welcome to Baml! To use this generated code, please run the following:
#
#  $ pip install baml-py
#
###############################################################################

# This file was generated by BAML: please do not edit it. Instead, edit the
# BAML files and re-generate this code.
#
# ruff: noqa: E501,F401
# flake8: noqa: E501,F401
# pylint: disable=unused-import,line-too-long
# fmt: off
import baml_py
from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeAlias
from typing import Dict, Generic, List, Literal, Optional, TypeVar, Union


T = TypeVar('T')
CheckName = TypeVar('CheckName', bound=str)

class Check(BaseModel):
    name: str
    expression: str
    status: str

class Checked(BaseModel, Generic[T,CheckName]):
    value: T
    checks: Dict[CheckName, Check]

def get_checks(checks: Dict[CheckName, Check]) -> List[Check]:
    return list(checks.values())

def all_succeeded(checks: Dict[CheckName, Check]) -> bool:
    return all(check.status == "succeeded" for check in get_checks(checks))



class EvaluationType(str, Enum):
    
    Definitive = "Definitive"
    Freshness = "Freshness"
    Plurality = "Plurality"
    Attribution = "Attribution"
    Completeness = "Completeness"

class KnowledgeType(str, Enum):
    
    QA = "QA"
    Strategy = "Strategy"
    SearchInfo = "SearchInfo"
    URL = "URL"

class ActionWithThink(BaseModel):
    model_config = ConfigDict(extra='allow')
    thinking: str
    action: Union["Search", "Answer", "Reflect", "Visit"]

class AlternativeSearchResult(BaseModel):
    title: str
    link: str
    snippet: Optional[str] = None

class Answer(BaseModel):
    references: Optional[List["Reference"]] = None
    answer_text: str

class AttributionAnalysis(BaseModel):
    think: str
    pass_evaluation: bool
    sources_provided: bool
    sources_verified: bool
    quotes_accurate: bool

class BadContext(BaseModel):
    question: str
    answer: str
    evaluation: str
    recap: str
    blame: str
    improvement: str

class CompletenessAnalysis(BaseModel):
    think: str
    pass_evaluation: bool
    aspects_expected: str
    aspects_provided: str

class DedupOutput(BaseModel):
    think: str
    unique_queries: List[str]

class DefinitiveAnalysis(BaseModel):
    think: str
    pass_evaluation: bool

class ErrorAnalysisOutput(BaseModel):
    recap: str
    blame: str
    improvement: str
    next_search: List[str]

class ExtractedSegments(BaseModel):
    segment_list: str

class FreshnessAnalysis(BaseModel):
    think: str
    pass_evaluation: bool
    days_ago: Optional[int] = None
    max_age_days: Optional[int] = None

class KnowledgeItem(BaseModel):
    question: str
    answer: str
    references: Union[List[Dict[str, str]], Optional[List[str]]] = None
    type: "KnowledgeType"
    updated: Optional[str] = None

class Passage(BaseModel):
    text: str
    query: str

class PluralityAnalysis(BaseModel):
    think: str
    pass_evaluation: bool
    count_expected: Optional[int] = None
    count_provided: Optional[int] = None

class QueryRewriterOutput(BaseModel):
    think: str
    queries: List[str]

class QuestionEvaluationOutput(BaseModel):
    think: str
    needsFreshness: bool
    needsPlurality: bool
    needsCompleteness: bool

class Reference(BaseModel):
    exactQuote: str
    url: str
    title: Optional[str] = None

class Reflect(BaseModel):
    questions_to_answer: List[str]

class ReportOutput(BaseModel):
    report: str

class Search(BaseModel):
    search_requests: List[str]

class StandardSearchResult(BaseModel):
    title: str
    url: str
    description: Optional[str] = None

class Visit(BaseModel):
    urls: List[str]

JsonObject: TypeAlias = Dict[str, "JsonValue"]

JsonValue: TypeAlias = Union[int, str, bool, float, "JsonObject"]
