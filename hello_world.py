from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
import asyncio

kernel = Kernel()

# configuring serivce
service_id = "chat-gpt"
kernel.add_service(
    OpenAIChatCompletion(
        service_id = service_id,
        ai_model_id="gpt-3.5-turbo",
    )
)

# configure request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.7
req_settings.top_p = 0.8

prompt = "Tell me a joke"

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="joke",
    template_format="semantic-kernel",
    execution_settings=req_settings
)

function = kernel.add_function(
    function_name="tell_joke_function",
    plugin_name="tell_joke_plugin",
    prompt_template_config=prompt_template_config
)

async def main():
    result = await kernel.invoke(function)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())