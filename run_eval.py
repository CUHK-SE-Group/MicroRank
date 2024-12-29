from pathlib import Path

import pandas as pd
import toml
from concurrent.futures import ProcessPoolExecutor

from online_rca import online_anomaly_detect_RCA
from preprocess_data import get_operation_slo, get_service_operation_list
import datetime
from aiomultiprocess import Pool

async def evaluate(case_dir:Path):
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
    normal_data['Timestamp'] = pd.to_datetime(normal_data['Timestamp'])
    abnormal_data['Timestamp'] = pd.to_datetime(abnormal_data['Timestamp'])
    span_df = normal_data.copy()
    operation_list = get_service_operation_list(span_df)
    operation_slo = get_operation_slo(operation_list, span_df)

    result = online_anomaly_detect_RCA(abnormal_data, operation_slo, operation_list)
    return case_dir.name, result

def calculate_metrics(prediction_data, gt_list):
    AC_at_1 = 0
    AC_at_3 = 0
    AC_at_5 = 0    
    avg_at_5 = 0
    total_cases = len(gt_list)


    for gt in gt_list:
        case_name = gt['case']
        gt_service = gt['service']
        
        # Find the corresponding case in prediction_data
        for case, top_list in prediction_data:
            if case == case_name:
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

    return {
        "AC@1": AC_at_1,
        "AC@3": AC_at_3,
        "AC@5": AC_at_5,
        "Avg@5": avg_at_5
    }
async def main():
    root_base_dir = Path("/home/nn/workspace/Metis-DataSet/ts-all")
    fault_injection_file = root_base_dir / "fault_injection.toml"
    data = toml.load(fault_injection_file)
    gt_list = data["chaos_injection"]

    case_dirs = [p for p in root_base_dir.iterdir() if p.is_dir()]
    async with Pool() as pool:
        prediction_data = list(await pool.map(evaluate,case_dirs))
        result = calculate_metrics(prediction_data, gt_list)
        # save as result.csv
        result_df = pd.DataFrame([result])
        result_df.to_csv("result.csv", index=False)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

            