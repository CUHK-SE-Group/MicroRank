import codecs
import csv
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.parser import parse

from anormaly_detector import system_anomaly_detect, trace_list_partition
from pagerank import trace_pagerank
from preprocess_data import (
    get_operation_duration_data,
    get_operation_slo,
    get_pagerank_graph,
    get_service_operation_list,
    get_span,
)


def timestamp(datetime):
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    # print(ts)
    return ts


def calculate_spectrum_without_delay_list(
    anomaly_result,
    normal_result,
    anomaly_list_len,
    normal_list_len,
    top_max,
    normal_num_list,
    anomaly_num_list,
    spectrum_method,
):
    spectrum = {}

    for node in anomaly_result:
        spectrum[node] = {}
        # spectrum[node]['ef'] = anomaly_result[node] * anomaly_list_len
        # spectrum[node]['nf'] = anomaly_list_len - anomaly_result[node] * anomaly_list_len
        spectrum[node]['ef'] = anomaly_result[node] * anomaly_num_list[node]
        spectrum[node]['nf'] = anomaly_result[node] * (anomaly_list_len - anomaly_num_list[node])
        if node in normal_result:
            # spectrum[node]['ep'] = normal_result[node] * normal_list_len
            # spectrum[node]['np'] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]['ep'] = normal_result[node] * normal_num_list[node]
            spectrum[node]['np'] = normal_result[node] * (normal_list_len - normal_num_list[node])
        else:
            spectrum[node]['ep'] = 0.0000001
            spectrum[node]['np'] = 0.0000001

    for node in normal_result:
        if node not in spectrum:
            spectrum[node] = {}
            # spectrum[node]['ep'] = normal_result[node] * normal_list_len
            # spectrum[node]['np'] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]['ep'] = (1 + normal_result[node]) * normal_num_list[node]
            spectrum[node]['np'] = normal_list_len - normal_num_list[node]
            if node not in anomaly_result:
                spectrum[node]['ef'] = 0.0000001
                spectrum[node]['nf'] = 0.0000001

    # print('\n Micro Rank Spectrum raw:')
    # print(json.dumps(spectrum))
    result = {}

    for node in spectrum:
        # Dstar2
        if spectrum_method == "dstar2":
            result[node] = spectrum[node]['ef'] * spectrum[node]['ef'] / (spectrum[node]['ep'] + spectrum[node]['nf'])
        # Ochiai
        elif spectrum_method == "ochiai":
            result[node] = spectrum[node]['ef'] / math.sqrt(
                (spectrum[node]['ep'] + spectrum[node]['ef']) * (spectrum[node]['ef'] + spectrum[node]['nf'])
            )

        elif spectrum_method == "jaccard":
            result[node] = spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['ep'] + spectrum[node]['nf'])

        elif spectrum_method == "sorensendice":
            result[node] = (
                2 * spectrum[node]['ef'] / (2 * spectrum[node]['ef'] + spectrum[node]['ep'] + spectrum[node]['nf'])
            )

        elif spectrum_method == "m1":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np']) / (spectrum[node]['ep'] + spectrum[node]['nf'])

        elif spectrum_method == "m2":
            result[node] = spectrum[node]['ef'] / (
                2 * spectrum[node]['ep'] + 2 * spectrum[node]['nf'] + spectrum[node]['ef'] + spectrum[node]['np']
            )
        elif spectrum_method == "goodman":
            result[node] = (2 * spectrum[node]['ef'] - spectrum[node]['nf'] - spectrum[node]['ep']) / (
                2 * spectrum[node]['ef'] + spectrum[node]['nf'] + spectrum[node]['ep']
            )
        # Tarantula
        elif spectrum_method == "tarantula":
            result[node] = (
                spectrum[node]['ef']
                / (spectrum[node]['ef'] + spectrum[node]['nf'])
                / (
                    spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['nf'])
                    + spectrum[node]['ep'] / (spectrum[node]['ep'] + spectrum[node]['np'])
                )
            )
        # RussellRao
        elif spectrum_method == "russellrao":
            result[node] = spectrum[node]['ef'] / (
                spectrum[node]['ef'] + spectrum[node]['nf'] + spectrum[node]['ep'] + spectrum[node]['np']
            )

        # Hamann
        elif spectrum_method == "hamann":
            result[node] = (
                spectrum[node]['ef'] + spectrum[node]['np'] - spectrum[node]['ep'] - spectrum[node]['nf']
            ) / (spectrum[node]['ef'] + spectrum[node]['nf'] + spectrum[node]['ep'] + spectrum[node]['np'])

        # Dice
        elif spectrum_method == "dice":
            result[node] = (
                2 * spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['nf'] + spectrum[node]['ep'])
            )

        # SimpleMatching
        elif spectrum_method == "simplematcing":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np']) / (
                spectrum[node]['ef'] + spectrum[node]['np'] + spectrum[node]['nf'] + spectrum[node]['ep']
            )

        # RogersTanimoto
        elif spectrum_method == "rogers":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np']) / (
                spectrum[node]['ef'] + spectrum[node]['np'] + 2 * spectrum[node]['nf'] + 2 * spectrum[node]['ep']
            )

    # Top-n节点列表
    top_list = []
    score_list = []
    for index, score in enumerate(sorted(result.items(), key=lambda x: x[1], reverse=True)):
        if index < top_max + 6:
            top_list.append(score[0])
            score_list.append(score[1])
            print('%-50s: %.8f' % (score[0], score[1]))
    return top_list, score_list


def online_anomaly_detect_RCA(data, slo, operation_list):

    # Define the time window
    window_duration_normal = pd.Timedelta(minutes=5)
    window_duration_abnormal = pd.Timedelta(minutes=4)
    # Iterate over each 1-minute window
    start = data['Timestamp'].min()
    end = data['Timestamp'].max()
    current_time = start
    while current_time < end:
        start_time = current_time
        end_time = current_time + window_duration_normal
        anomaly_flag, normal_list, abnormal_list = system_anomaly_detect(
            data, start_time=start_time, end_time=end_time, slo=slo, operation_list=operation_list
        )
        if anomaly_flag:

            print("anomaly_list", len(abnormal_list))
            print("normal_list", len(normal_list))
            print("total", len(normal_list) + len(abnormal_list))

            if not abnormal_list or not normal_list:
                current_time += window_duration_normal
                continue

            operation_operation, operation_trace, trace_operation, pr_trace = get_pagerank_graph(normal_list, data)
            normal_trace_result, normal_num_list = trace_pagerank(
                operation_operation, operation_trace, trace_operation, pr_trace, False
            )

            a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace = get_pagerank_graph(
                abnormal_list, data
            )
            anomaly_trace_result, anomaly_num_list = trace_pagerank(
                a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace, True
            )

            top_list, score_list = calculate_spectrum_without_delay_list(
                anomaly_result=anomaly_trace_result,
                normal_result=normal_trace_result,
                anomaly_list_len=len(abnormal_list),
                normal_list_len=len(normal_list),
                top_max=15,
                anomaly_num_list=anomaly_num_list,
                normal_num_list=normal_num_list,
                spectrum_method="ochiai",
            )
            print(top_list, score_list)
            # Pair services with scores
            service = {}
            for span, score in zip(top_list, score_list):
                if span.split('_')[0] not in service:
                    service[span.split('_')[0]] = 0
                service[span.split('_')[0]] += score
            #paired = list(zip(top_list, score_list))

            # Sort by confidence descending
            sorted_paired = sorted(service.items(), key=lambda x: x[1], reverse=True)
            
            return [item[0].split('_')[0] for item in sorted_paired]
            current_time += window_duration_abnormal  # + extra 4min
        current_time += window_duration_normal  # + 1min
    return []


if __name__ == '__main__':
    base_dir = Path("/home/nn/workspace/MicroRank/ts11-22/seat-1130-1558")
    normal_data = pd.read_csv(base_dir / "normal" / "traces.csv").rename(
        columns={
            'TraceId': 'traceID',
            "ServiceName": "serviceName",
            "SpanName": "operationName",
            "PodName": "podName",
            'SpanId': 'spanID',
            'Duration': 'duration',
            "TraceStart": "startTime",
            "TraceEnd": "endTime",
        }
    )
    abnormal_data = pd.read_csv(base_dir / "abnormal" / "traces.csv").rename(
        columns={
            'TraceId': 'traceID',
            "ServiceName": "serviceName",
            "SpanName": "operationName",
            "PodName": "podName",
            'SpanId': 'spanID',
            'Duration': 'duration',
            "TraceStart": "startTime",
            "TraceEnd": "endTime",
        }
    )
    normal_data['startTime'] = pd.to_datetime(normal_data['startTime'])
    normal_data['endTime'] = pd.to_datetime(normal_data['endTime'])
    abnormal_data['startTime'] = pd.to_datetime(abnormal_data['startTime'])
    abnormal_data['endTime'] = pd.to_datetime(abnormal_data['endTime'])
    span_df = normal_data.copy()
    # print(span_list)
    operation_list = get_service_operation_list(span_df)
    print(operation_list)
    operation_slo = get_operation_slo(operation_list, span_df)

    online_anomaly_detect_RCA(abnormal_data, operation_slo, operation_list)
