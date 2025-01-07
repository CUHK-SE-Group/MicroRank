import datetime
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import toml
from online_rca import online_anomaly_detect_RCA
from preprocess_data import get_operation_slo, get_service_operation_list


def evaluate(case_dir: Path):
    print(f"Processing {case_dir}")
    normal_data = pd.read_csv(case_dir / "normal" / "traces.csv").rename(
        columns={
            'TraceId': 'traceID',
            "ServiceName": "serviceName",
            "SpanName": "operationName",
            'SpanId': 'spanID',
            'Duration': 'duration',
        }
    )
    abnormal_data = pd.read_csv(case_dir / "abnormal" / "traces.csv").rename(
        columns={
            'TraceId': 'traceID',
            "ServiceName": "serviceName",
            "SpanName": "operationName",
            'SpanId': 'spanID',
            'Duration': 'duration',
        }
    )
    normal_data['Timestamp'] = pd.to_datetime(normal_data['Timestamp'], format='mixed')
    abnormal_data['Timestamp'] = pd.to_datetime(abnormal_data['Timestamp'], format='mixed')
    span_df = normal_data.copy()
    operation_list = get_service_operation_list(span_df)
    operation_slo = get_operation_slo(operation_list, span_df)

    result = online_anomaly_detect_RCA(abnormal_data, operation_slo, operation_list)
    return case_dir.name, result


def calculate_overlap(start1, end1, start2, end2):
    if start1 is None or end1 is None or start2 is None or end2 is None:
        return False  # No overlap if any input is None
    return max(start1, start2) < min(end1, end2)


def f1_score(predicted_range, ground_truth_range):
    """
    Calculate the F1 score based on predicted and ground truth time ranges.

    Args:
        predicted_range (tuple): A tuple with predicted start and end times (datetime, datetime).
        ground_truth_range (tuple): A tuple with ground truth start and end times (datetime, datetime).

    Returns:
        float: precision, recall, F1 score.
    """
    predicted_start, predicted_end = predicted_range
    truth_start, truth_end = ground_truth_range
    predicted_start, predicted_end, truth_start, truth_end = map(
        lambda x: int(pd.to_datetime(x).as_unit("s").timestamp()),
        [predicted_start, predicted_end, truth_start, truth_end],
    )
    # Calculate overlap (True Positive)
    if calculate_overlap(predicted_start, predicted_end, truth_start, truth_end):
        tp = 1
        fp = 0
        fn = 0
    else:
        tp = 0
        fp = 1  # No overlap, predicted range is considered as false positive
        fn = 1  # No overlap, ground truth range is considered as false negative

    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1 score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0  # If precision and recall are both 0, F1 is 0

    return precision, recall, f1


def calculate_metrics(prediction_data, gt_list):
    AC_at_1 = 0
    AC_at_3 = 0
    AC_at_5 = 0
    avg_at_5 = 0
    total_cases = len(gt_list)
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for gt in gt_list:
        case_name = gt['case']
        gt_service = gt['service']
        # Calculate gt_abnormal_time_range
        gt_start_time = gt['timestamp']
        gt_end_time = gt_start_time + timedelta(minutes=5)
        gt_abnormal_time_range = (gt_start_time, gt_end_time)

        # Find the corresponding case in prediction_data
        for case, ((predicted_start, predicted_end), top_list) in prediction_data:
            if case == case_name:
                # Calculate F1 score using predict_abnormal_time_range and gt_abnormal_time_range
                if not len(top_list):
                    # No anomlies predicted
                    precision, recall, f1 = 0, 0, 0
                    continue
                else:
                    precision, recall, f1 = f1_score((predicted_start, predicted_end), gt_abnormal_time_range)

                # Update total precision, recall, and f1
                total_precision += precision
                total_recall += recall
                total_f1 += f1

                # Find ground truth service index in the combined_ranking dataframe
                if gt_service in top_list:
                    service_index = top_list.index(gt_service) + 1

                    # AC@1
                    if service_index == 1:
                        AC_at_1 += 1

                    # AC@3
                    if service_index <= 3:
                        AC_at_3 += 1

                    # AC@5
                    if service_index <= 5:
                        AC_at_5 += 1

                    # Avg@5
                    if service_index <= 5:
                        avg_at_5 += (5 - service_index + 1) / 5.0

                break

    # Ranking metrics
    AC_at_1 /= total_cases
    AC_at_3 /= total_cases
    AC_at_5 /= total_cases
    avg_at_5 /= total_cases
    precision_avg = total_precision / total_cases
    recall_avg = total_recall / total_cases
    f1_avg = total_f1 / total_cases

    return {
        "AD Precision": precision_avg,
        "AD Recall": recall_avg,
        "AD F1": f1_avg,
        "AC@1": AC_at_1,
        "AC@3": AC_at_3,
        "AC@5": AC_at_5,
        "Avg@5": avg_at_5,
    }


def main():
    root_base_dir = Path("/home/nn/workspace/Metis-DataSet/ts-all")
    fault_injection_file = root_base_dir / "fault_injection.toml"
    data = toml.load(fault_injection_file)
    gt_list = data["chaos_injection"]

    case_dirs = [p for p in root_base_dir.iterdir() if p.is_dir()]
    with Pool() as pool:
        prediction_data = list(pool.map(evaluate, case_dirs))
        result = calculate_metrics(prediction_data, gt_list)
        # save as result.csv
        result_df = pd.DataFrame([result])
        result_df.to_csv("result.csv", index=False)


if __name__ == "__main__":
    main()
