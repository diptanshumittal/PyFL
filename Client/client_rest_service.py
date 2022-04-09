import multiprocessing
import uuid
import threading
from flask import Flask, jsonify, request
import json
import time
import logging
from flask_autodoc.autodoc import Autodoc


# from flask.ext.autodoc import Autodoc


class ClientRestService:

    def __init__(self, config):
        print(config)
        self.model_trainer = config["model_trainer"]
        self.minio_client = config["minio_client"]
        self.port = config['flask_port']
        self.server = config["server"]
        self.global_model_path = config["global_model_path"]
        self.stop_round_event = threading.Event()
        # self.round_thread = None
        self.round_process = None
        self.config = {"round_id": -1}

    def run(self):
        # log = logging.getLogger('werkzeug')
        # log.setLevel(logging.ERROR)
        app = Flask(__name__)
        auto = Autodoc(app)

        @app.route('/documentation')
        @auto.doc()
        def documentation():
            """return API documentation page"""
            return auto.html()

        @app.route('/')
        @auto.doc()
        def index():
            """return the type of flask server"""
            return jsonify({'description': "This is the client"})

        @app.route('/startround')
        @auto.doc()
        def start_round():
            """Used by the reducer to request client to start a round as per the given parameters"""
            self.config = {
                "round_id": request.args.get('round_id', None),
                "bucket_name": request.args.get('bucket_name', None),
                "global_model": request.args.get('global_model', None),
                "epochs": int(request.args.get('epochs', "1"))
            }
            self.round_process = multiprocessing.Process(target=self.run_round, args=(self.config,))
            self.round_process.start()
            return jsonify({'status': "started"})

        @app.route('/stopround')
        @auto.doc()
        def stop_round():
            """Used by the reducer to request client to stop the ongoing round"""
            multiprocessing.Process(target=self.stop_round).start()
            return jsonify({'status': "stopping"})

        app.run(host="0.0.0.0", port=self.port)

    def stop_round(self):
        print("Stop requested by the server", flush=True)
        self.round_process.terminate()
        self.server.send_round_stop_request(self.config["round_id"])

    def run_round(self, config):
        """
        Function to train the global model locally using local dataset, evaluate its performance and send the
        statistics, trained model back to the reducer.
        :param config: The hyperparameters for a given round.
        :return:
        """
        try:
            pre = time.time()
            print("Running round - ", config["round_id"], flush=True)
            self.minio_client.fget_object('fedn-context', config["global_model"], self.global_model_path)
            report = self.model_trainer.start_round({"epochs": config["epochs"]})
            if report["status"] != "fail":
                self.minio_client.fput_object(config["bucket_name"], str(uuid.uuid4()) + ".npz", self.global_model_path)
            report["round_time"] = time.time() - pre
            print("Report : ", report, flush=True)
            if self.server.send_round_complete_request(config["round_id"], json.dumps(report)):
                return
            print("Round notification not sent to reducer successfully!!!", flush=True)
        except Exception as e:
            print("Error during round :", e)
