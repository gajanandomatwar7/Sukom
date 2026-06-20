from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate


class Department(str, Enum):
    SANITATION = "Sanitation"
    ROAD_INFRA = "Road Infrastructure"
    GARBAGE = "Garbage Management"
    WATER_LOGGING = "Water Logging & Drainage"
    ELECTRICITY = "Electricity"
    WATER_SUPPLY = "Water Supply"
    STREET_LIGHT = "Street Lighting"
    SEWERAGE = "Sewerage"
    PARKS = "Parks & Gardens"
    OTHER = "Other"


class DepartmentScore(BaseModel):
    department: Department
    confidence: float = Field(ge=0, le=1)


class ComplaintCategory(BaseModel):
    departments: List[DepartmentScore] = Field(
        description=(
            "List all municipal departments whose confidence is greater than 0.90."
        )
    )

    reason: str = Field(
        description="Brief explanation for why these departments were selected."
    )


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

structured_llm = llm.with_structured_output(ComplaintCategory)


departments = "\n".join(f"- {d.value}" for d in Department)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
You are a municipal complaint classifier.

Your job is to identify ALL relevant municipal departments.

Available departments:
{departments}

For each selected department:
- assign a confidence score between 0 and 1.
- include ONLY departments whose confidence is >= 0.90.
- If none reach 0.90, return only the single best matching department.

Return only the structured output.
""",
        ),
        ("human", "{complaint}"),
    ]
)

chain = prompt | structured_llm


result = chain.invoke(
    {
        "complaint": "rastyat khup pani sachlay, khadyan mule traffic jam hot ahe"
    }
)


for dept in result.departments:
    print(
        dept.department.value,
    )


print('---------------------------')

departments = [
    d.department.value
    for d in result.departments
]

print(departments)