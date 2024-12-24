# Based on https://tree-sitter.github.io/tree-sitter/code-navigation-systems?ref=blog.lancedb.com#tagging-and-captures

from enum import StrEnum
from typing import List

from commons import FilepathType
from pydantic import BaseModel, Field


class TagRole(StrEnum):
    """Represents the role of an entity in the code."""

    DEFINITION = "definition"
    REFERENCE = "reference"


class TagKind(StrEnum):
    """Represents the kind of entity in the code."""

    CLASS = "class"
    FUNCTION = "function"
    CALL = "call"


class Tag(BaseModel):
    """
    Represents a tag extracted from source code using tree-sitter.

    Follows the @role.kind capture name format:
    - role: definition or reference
    - kind: class, function, method, call, etc.
    """

    name: str = Field(..., description="Name of the identified entity")
    role: TagRole = Field(..., description="Role of the entity (definition/reference)")
    kind: TagKind = Field(..., description="Kind of the entity")
    start_line: int = Field(
        ..., description="The line where the identifier is defined or referenced"
    )
    rel_filename: str = Field(..., description="Relative path to the source file")
    abs_filename: str = Field(..., description="Full path to the source file")


class Define(BaseModel):
    """Track which files define a specific identifier"""

    identifier: str
    filepath: FilepathType


class Reference(BaseModel):
    """Track which files reference a specific identifier"""

    identifier: str
    filepaths: List[FilepathType]


class EntityRelationship(BaseModel):
    """Connect a define to its reference"""

    define: Define
    reference: Reference
