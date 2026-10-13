"""Script para testar o tratamento de dados com Pydantic."""
import json
from pathlib import Path
from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDefinition, TestCase, ToolCall


def test_data_pipeline():
    print("--- 1. TESTAR FICHEIROS REAIS ---")
    functions_file = Path("data/input/functions_definition.json")
    tests_file = Path("data/input/function_calling_tests.json")

    # Validar functions_definition.json se existir
    if functions_file.exists():
        try:
            with open(functions_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            # TypeAdapter valida uma lista de FunctionDefinition
            adapter = TypeAdapter(list[FunctionDefinition])
            functions = adapter.validate_python(raw_data)
            print(f"✅ functions_definition.json válido! ({len(functions)} funções carregadas)")
            print(f"   Exemplo carregado: {functions[0].name} -> parâmetros: {list(functions[0].parameters.keys())}")
        except ValidationError as e:
            print(f"❌ Erro de validação Pydantic no ficheiro real: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON inválido no ficheiro real: {e}")
    else:
        print(f"⚠️ Ficheiro não encontrado: {functions_file}")

    # Validar function_calling_tests.json se existir
    if tests_file.exists():
        try:
            with open(tests_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            adapter = TypeAdapter(list[TestCase])
            tests = adapter.validate_python(raw_data)
            print(f"✅ function_calling_tests.json válido! ({len(tests)} testes carregados)")
            print(f"   Primeiro prompt: '{tests[0].prompt}'")
        except Exception as e:
            print(f"❌ Erro ao ler testes reais: {e}")
    else:
        print(f"⚠️ Ficheiro não encontrado: {tests_file}")

    print("\n--- 2. TESTAR ROBUSTEZ CONTRA DADOS INVÁLIDOS ---")
    # Caso A: Tipo de parâmetro não permitido (ex: "list" quando só permitimos string/number/etc)
    bad_json = """
    [
        {
            "name": "fn_broken",
            "description": "Função com tipo inválido",
            "parameters": {
                "items": {"type": "array_invalido"}
            }
        }
    ]
    """
    try:
        data = json.loads(bad_json)
        TypeAdapter(list[FunctionDefinition]).validate_python(data)
        print("❌ Falhou: Deveria ter rejeitado o tipo inválido!")
    except ValidationError:
        print("✅ Sucesso: Pydantic barrou o tipo de parâmetro não reconhecido!")

    # Caso B: Saída gerada pela LLM (ToolCall)
    valid_tool_call = '{"name": "fn_add_numbers", "arguments": {"a": 40, "b": 2}}'
    call = ToolCall.model_validate_json(valid_tool_call)
    print(f"✅ ToolCall validado com sucesso: {call.name}({call.arguments})")


if __name__ == "__main__":
    test_data_pipeline()