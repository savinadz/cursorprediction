# cursorprediction

- Create + activate virtual env: `python3 -m venv venv` - `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Sync AWS cursor data: `make sync` (you need AWS credentials and access rights for this)

Adjust the `config.py` and perform experiments by running `python main.py`.
Many functions have a `verbose` flag that can be set to `true` if you need detailed information.