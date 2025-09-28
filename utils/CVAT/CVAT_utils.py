from cvat_sdk import make_client
from pyaml_env import parse_config, BaseConfig

config = BaseConfig(parse_config('../../configs/project_config.yaml'))

CVAT_URL = config.CVAT.url
USERNAME = config.CVAT.username
PASSWORD = config.CVAT.password


def find_file_in_cvat(client, filename: str):
    matches = []

    # Search in projects → tasks
    for project in client.projects.list():
        project_id = project.id
        # Low-level API call to list tasks in the project
        try:
            project_tasks, _ = client.api_client.tasks_api.list(project_id=project_id)
        except Exception as e:
            print(f"⚠️ Could not list tasks for project {project_id}: {e}")
            continue

        for task in project_tasks.get("results", []):
            task_id = task["id"]
            try:
                data_meta, _ = client.api_client.tasks_api.retrieve_data_meta(task_id)
            except Exception as e:
                print(f"⚠️ Could not retrieve data meta for task {task_id}: {e}")
                continue

            for idx, frame in enumerate(data_meta.get("frames", [])):
                frame_name = frame.get("name", "")
                if filename in frame_name:
                    matches.append({
                        "project": project_id,
                        "task": task_id,
                        "frame_index": idx,
                        "frame_name": frame_name,
                    })
                    break  # Stop after first match in this task

    return matches


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Find file by name in all projects/tasks")

    choice = input("Enter 1 or 2:  ").strip()

    with make_client(CVAT_URL) as client:
        client.login((USERNAME, PASSWORD))

        if choice == "1":
            filename = input("Enter filename to search: ").strip()
            results = find_file_in_cvat(client, filename)
            if results:
                for match in results:
                    print(
                        f"✅ File '{match['frame_name']}' found in Project {match['project']}, "
                        f"Task {match['task']} at frame index {match['frame_index']}")
            else:
                print(f"❌ File '{filename}' not found in any project/task.")


        else:
            exit()