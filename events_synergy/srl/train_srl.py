# from pathlib import Path
#
# from datasets import load_dataset
# from typer import echo, Option, Typer
# from typing import List
#
# from ..trainers.multi_task_trainer import trainer_seq2seq_multi
# from ..utils.helpers import parse_kv
#
# app = Typer()
#
#
# @app.command()
# def train(
#     config_file: Path,
#     dataset_names: List[str],
#     kv: str = Option(
#         None,
#         "--kv",
#         help="Key-value pairs, separated by commas, e.g., key1=value1,key2=value2",
#     ),
# ):
#     """
#     Train datasets of the form:
#         {
#             "prompt": "SRL for [predicate]: sentence with [predicate]",
#             "response": ARG-0: [arg0] | ARG-1: [arg1] | ... | ARG-N: [argn]
#         }
#     :param config_file:
#     :param dataset_names:
#     :param kv: override config file parameters with this. e.g., "num_train_epochs=20,per_device_train_batch_size=8"
#     :return:
#     """
#     dataset_names = list(set(dataset_names))
#     srl_dataset_dict = {}
#
#     for ds_name in dataset_names:
#         srl_dataset_dict[ds_name] = load_dataset(ds_name)
#
#     kv_dict = {}
#
#     if kv:
#         try:
#             kv_dict = parse_kv(kv)
#             echo("Received key-value arguments:")
#             for key, value in kv_dict.items():
#                 echo(f"{key}: {value}")
#         except ValueError as e:
#             echo(f"Error: {e}")
#     else:
#         echo("No key-value arguments provided.")
#
#     trainer_seq2seq_multi(
#         config_file,
#         srl_dataset_dict,
#         **kv_dict
#     )
#
#
# if __name__ == "__main__":
#     app()
