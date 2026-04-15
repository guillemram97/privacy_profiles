from privacy_evals import command_line_parser, manage_data
from privacy_evals.utils import utils
from privacy_evals.local_models import process_outputs
from argparse import Namespace
import pdb

def main(args: Namespace) -> None:
    if args.model_name == "gpt-4o-mini" or args.model_name == "gpt-4o":
        from privacy_evals.external_models import ExternalModel as EM
        local_model = False
        total_cost = 0
        llm = EM.ExternalModel(args.model_name, args.agent_type, args.temperature, args.prompt_variation)
    else:
        from privacy_evals.local_models import LocalModel as LM
        local_model = True
        llm = LM.LocalModel(args.model_name, args.agent_type, args.temperature, args.device, args.prompt_variation)

    id_experiment, config_experiment = utils.write_id_experiment(
        args.id_experiment,
        args.agent_type,
        args.outputs_dir,
        args.model_name,
        llm.template_prompt,
        args.persona,
        args.prompt_variation,
    )
    data = manage_data.load_data(
        args.num_datapoints,
        args.num_splits,
        args.n_init,
        args.agent_type,
        id_experiment,
        args.outputs_dir,
        args.model_name,
        args.persona,
        args.model_answer_post,
        config_experiment,
    )
    data = llm.make_prompts(data)
    for idx in range(len(data)):
        if idx % 100 == 0:
            id_tmp, response_tmp, _ = manage_data.load_response(
                args.outputs_dir, args.model_name, args.agent_type, id_experiment
            )

        idx_wildchat = int(data.iloc[idx]["idx"])
        if (type(id_tmp) == int and id_tmp == -1) or not idx_wildchat in id_tmp:
            inputs = llm.datapoint_to_inputs(data.iloc[idx]["prompt"])
            if local_model:
                response = llm.generate_response(inputs)
                price = 0
            else:
                response, price = llm.generate_response(inputs)
                total_cost += price
            response = process_outputs.process_output(response, model_name=args.model_name)
            
            manage_data.save_output(
                args.outputs_dir,
                idx_wildchat,
                response,
                data.iloc[idx]["prompt"],
                args.model_name,
                args.agent_type,
                id_experiment,
                price,
            )
    if not local_model:
        # Save the total cost to a file
        utils.update_cost_experiment(
            id_experiment,
            args.agent_type,
            args.model_name,
            llm.template_prompt,
            args.outputs_dir,
            total_cost,
    )
    print('FINALISED')

if __name__ == "__main__":
    args = command_line_parser.parse_args()
    main(args)
