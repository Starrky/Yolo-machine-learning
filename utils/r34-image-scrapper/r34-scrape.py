from rule34Py import rule34Py
import requests
from pathlib import Path
from pyaml_env import parse_config, BaseConfig

config = BaseConfig(parse_config('../../configs/project_config.yaml'))

client = rule34Py()

client.api_key = config.utils.r34_api_key
client.user_id = config.utils.r34_user_id

DOWNLOAD_LIMIT = 500
add_tags = [""]

# Some weird stuff filter: use -something to remove it from search query
TAGS=[ "sort:score", "-video"
      , "-anthro", "-lizard_girl", "-furry", "-loli",  "-extremely_long_penis",
      "-scat", "-guro", "-vore",  "-absurdly_large_cock", "-comic",
      "-huge_cock", "-huge_breasts", "-huge_belly", "-huge_ass", "-pregnant",
      "-age_difference", "-teen", "-large_testicles", "-large_balls",
      "-torture"]

for element in add_tags:
    TAGS.append(element)

DOWNLOAD_DIR = Path(f"images/{add_tags[0]}")  # choose your directory
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)  # make sure it exists

str_list = list(map(str, TAGS))
delimiter_plus = "+"
print(f"Link is: https://rule34.xxx/index.php?page=post&s=list&tags={delimiter_plus.join(str_list)}")

results = client.search(tags=TAGS)


for result in results[0:DOWNLOAD_LIMIT]:
    print(f"Downloading post {result.id} ({result.image}).")

    # Build full path: directory + filename
    output_path = DOWNLOAD_DIR / Path(result.image).name
    try:
        with open(output_path, "wb") as fp_output:
            resp = requests.get(result.image)
            resp.raise_for_status()
            fp_output.write(resp.content)

    except Exception as e:
        print(f"Failed to download {result.id}: {e}")
