sync:
	AWS_PROFILE=aoeplay aws s3 sync s3://aoelake/cursortracker data

experiments:
	python main.py

