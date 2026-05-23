"""Graph schema data models.

Vendored from cypherbench.schema to keep this repo self-contained.
"""

from __future__ import annotations

import copy
import json
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class TemplateInfo(BaseModel):
    match_category: str
    match_cypher: str
    return_pattern_id: str
    return_cypher: str


class Nl2CypherSample(BaseModel):
    qid: str
    graph: str
    gold_cypher: str
    gold_match_cypher: Optional[str] = None
    nl_question: Optional[str] = None
    nl_question_raw: Optional[str] = None
    answer_json: Optional[str] = None
    from_template: TemplateInfo
    pred_cypher: Optional[str] = None
    metrics: Dict[str, float] = {}


class DataType(Enum):
    CATEGORICAL = "categorical"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    STR_ARRAY = "list[str]"

    @classmethod
    def from_neo4j_type(cls, t: str) -> DataType:
        mapping = {
            "STRING": cls.STR,
            "INTEGER": cls.INT,
            "FLOAT": cls.FLOAT,
            "BOOLEAN": cls.BOOL,
            "DATE": cls.DATE,
            "LIST": cls.STR_ARRAY,
            "LIST OF STRING": cls.STR_ARRAY,
        }
        return mapping[t]

    @classmethod
    def from_simplekg_type(cls, t: str) -> DataType:
        mapping = {
            "str": cls.STR,
            "int": cls.INT,
            "float": cls.FLOAT,
            "bool": cls.BOOL,
            "date": cls.DATE,
            "list[str]": cls.STR_ARRAY,
        }
        return mapping[t]


class EntitySchema(BaseModel):
    label: str
    description: Optional[str] = None
    properties: dict[str, DataType]


class RelationSchema(BaseModel):
    label: str
    subj_label: str
    obj_label: str
    properties: dict[str, DataType]


class PropertyGraphSchema(BaseModel):
    name: str
    entities: list[EntitySchema]
    relations: list[RelationSchema]

    def to_json(self, exclude_description: bool = False) -> dict:
        res = self.model_dump(mode="json")
        if exclude_description:
            for ent in res["entities"]:
                ent.pop("description", None)
        return res

    def to_str(self, exclude_description: bool = False) -> str:
        return json.dumps(self.to_json(exclude_description), indent=2)

    def to_sorted(self) -> PropertyGraphSchema:
        schema = copy.deepcopy(self)
        schema.entities = sorted(schema.entities, key=lambda x: x.label)
        schema.relations = sorted(schema.relations, key=lambda x: (x.label, x.subj_label, x.obj_label))
        for x in schema.entities + schema.relations:
            x.properties = dict(sorted(x.properties.items()))
        return PropertyGraphSchema(**schema.model_dump(mode="json"))

    @classmethod
    def from_json(cls, data: dict, add_meta_properties: dict | None = None) -> PropertyGraphSchema:
        if add_meta_properties is None:
            add_meta_properties = {"name": DataType.STR}
        schema = cls(**data)
        for ent in schema.entities:
            if add_meta_properties:
                ent.properties = dict(**add_meta_properties, **ent.properties)
        schema = cls(**schema.model_dump(mode="json"))
        return schema


class RelationInfo(BaseModel):
    label: str
    variants: List[str]
    is_symmetric: bool
    is_time_sensitive: bool
    is_mandatory_subj: bool
    is_mandatory_obj: bool
    subj_cardinality: Literal["one", "many"]
    obj_cardinality: Literal["one", "many"]
    implied_relations: List[str] = []


class GraphInfo(BaseModel):
    relations: List[RelationInfo]
