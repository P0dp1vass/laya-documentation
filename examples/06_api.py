"""Run the real HTTP service. --help does not load ML libraries or model weights."""
import argparse
import os

from production.runtime import add_runtime_arguments, settings_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = settings_from_args(args)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    for field, value in vars(settings).items():
        key = "LAYA_" + field.upper()
        if value is not None:
            os.environ[key] = str(value)
    import uvicorn
    uvicorn.run("production.api:app", host=args.host, port=args.port, workers=1, limit_concurrency=32)


if __name__ == "__main__":
    main()
