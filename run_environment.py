import os
import json
import logging
import time
import yaml

from langchain.requests import Requests
from langchain_community.chat_models import ChatOpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()


def main():
    config = yaml.load(open('keys_secret.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ["OPENAQ_API_KEY"] = config['openaq_api_key']

    query_idx = 1

    log_dir = os.path.join("logs", "restgpt_environment")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint()), logging.FileHandler(os.path.join(log_dir, f"{query_idx}.log"), mode='w', encoding='utf-8')],
    )
    logger.setLevel(logging.INFO)

    with open("specs/environment.json") as f:
        raw_api_spec = json.load(f)

    api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

    headers = {
        "X-API-Key": os.environ["OPENAQ_API_KEY"]
    }

    requests_wrapper = Requests(headers=headers)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0)
    rest_gpt = RestGPT(llm,
                       api_spec=api_spec,
                       scenario='spotify',
                       requests_wrapper=requests_wrapper,
                       simple_parser=False)

    start_time = time.time()
    rest_gpt.run("What's pollution like in R K Puram, Delhi?")
    logger.info(f"Execution Time: {time.time() - start_time}")


if __name__ == '__main__':
    main()
