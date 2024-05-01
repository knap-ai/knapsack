import json
import re
from functools import partial

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range


def sql_where_to_qdrant_json(sql_where_clause: str):
    def handle_comparison(match):
        field, op, value = match.groups()
        if op == '=':
            return f'{{"key": "{field}", "match": {value}}}'
        elif op in ['>', '>=', '<', '<=']:
            if op == '>':
                qdrant_op = 'gt'
            elif op == '>=':
                qdrant_op = 'gte'
            elif op == '<':
                qdrant_op = 'lt'
            elif op == '<=':
                qdrant_op = 'lte'
            else: 
                raise ValueError(f"Converting SQL to Qdrant Filter. Encountered bad value: {op}")
            return f'{{"key": "{field}", "range": {{"{qdrant_op}": {value}}}}}'
    
    def handle_is_null(match):
        field, is_not = match.groups()
        return f'{{"key": "{field}", "is_none": {bool(is_not)}}}'

    # Replace logical operators (AND, OR, NOT)
    sql_where_clause = _sql_where_replacements(sql_where_clause)

    # Replace comparisons (>, >=, <, <=, =)
    sql_where_clause = re.sub(r'(["\']?\w+["\']?)\s*(=|>|>=|<|<=)\s*(["\']?\w+["\']?)', handle_comparison, sql_where_clause)

    # Replace IS NULL and IS NOT NULL
    sql_where_clause = re.sub(r'(\w+)\s+IS\s+(NOT\s+)?NULL', handle_is_null, sql_where_clause)
    sql_where_clause = f'[{sql_where_clause}]'
    return sql_where_clause


def qdrant_json_to_filter(qdrant_filter: str) -> Filter:
    def parse_condition(condition):
        if 'key' in condition:
            key = condition['key']
            if 'match' in condition:
                return FieldCondition(key=key, match=MatchValue(value=condition['match']))
            elif 'range' in condition:
                range_cond = condition['range']
                return FieldCondition(key=key, range=Range(**range_cond))
            # elif 'is_none' in condition:
            #     return FieldCondition(key=key, is_none=condition['is_none'])
        elif 'must_not' in condition:
            return Filter(must_not=[parse_condition(cond) for cond in condition['must_not']])
        elif 'should' in condition:
            return Filter(should=[parse_condition(cond) for cond in condition['should']])
        else:
            raise ValueError("Unknown condition type")

    qdrant_filter_json = json.loads(qdrant_filter)
    filter_conditions = [parse_condition(cond) for cond in qdrant_filter_json]

    # Assuming all conditions are at the top level are 'must' conditions
    return Filter(must=filter_conditions)


def _sql_where_replacements(sql_where_clause) -> str:
    # Handle AND and NOT operators as before
    sql_where_clause = sql_where_clause.replace('AND', ',')
    sql_where_clause = sql_where_clause.replace('NOT', '{"must_not":')

    # Handle the OR operator with a regex
    # This simple regex assumes the OR operator has simple conditions on both sides
    # and may not correctly handle nested or complex expressions
    def handle_or(match):
        full_expression = match.group(0)
        # Remove the leading and trailing parentheses
        if full_expression[0] == '(' and full_expression[-1] == ')':
            inner_expression = full_expression[1:-1]
        else:
            inner_expression = full_expression
        # Split by 'OR', considering potential spaces around it
        parts = [part.strip() for part in inner_expression.split('OR')]
        # Transform parts into Qdrant JSON structure
        json_parts = ', '.join(parts)
        return f'{{"should": [{json_parts}]}}'

    # Use a regex to find and replace OR conditions
    # The following regex pattern is very simplistic and may need to be refined for more complex scenarios
    sql_where_clause = re.sub(r'\(([^()]*? OR [^()]*?)\)', handle_or, sql_where_clause)

    # Handle parentheses
    sql_where_clause = sql_where_clause.replace("(", "[")
    sql_where_clause = sql_where_clause.replace(")", "]")
    sql_where_clause = sql_where_clause.replace("'", "\"")

    return sql_where_clause


def convert_sql_where_to_qdrant_filter(filter: str | None) -> Filter | None:
    if filter is None:
        return None
    qdrant_json = sql_where_to_qdrant_json(filter)
    return qdrant_json_to_filter(qdrant_json)


if __name__ == "__main__":
    test_sql_where = "rank < 100000"
    print(f"SQL WHERE AT START: {test_sql_where}")
    test_qdrant_filter = sql_where_to_qdrant_json(test_sql_where)
    print(f"SQL WHERE -> QDRANT FILTER: {test_qdrant_filter}")
    print(f"first final result: {qdrant_json_to_filter(test_qdrant_filter)}")

    test_sql_complex = "age > 30 AND (salary >= 50000 OR department = 'Sales')"
    qdrant_filter_complex = sql_where_to_qdrant_json(test_sql_complex)
    # should look like this: '[{"key": "age", "range": {"gt": 30}}, {"should": [{"key": "salary", "range": {"gte": 50000}}, {"key": "department", "match": "Sales"}]}]'
    print(f"COMPLEX SQL TO QDRANT JSON FILTER: {qdrant_filter_complex}")
    qdrant_query = qdrant_json_to_filter(qdrant_filter_complex)
    print(f"second final result: {qdrant_query}")
