import dsp
import dspy
from dspy.signatures.signature import ensure_signature
from dspy.primitives.program import Module
from dspy.predict import Predict

# TODO: Simplify a lot.
# TODO: Divide Action and Action Input like langchain does for ReAct.

# Reflexion module based off of DSPy's ReAct

class Reflexion(Module):
    def __init__(
            self, 
            signature, 
            max_iters, 
            reflect_interval, 
            tools, 
            read_memory_tool=None, 
            write_memory_tool=None, 
            update_valid_actions_tool=None,
            generate_candidate_actions_tool=None,
            debug=False
            ):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.reflect_interval = reflect_interval
        self.debug = debug
        self.last_printed_output_idx = 0
        
        self.oracle_insertions = {}

        self.tools = tools 
        self.tools = {tool.name: tool for tool in self.tools}
        self.read_memory_tool = read_memory_tool
        self.write_memory_tool = write_memory_tool
        self.update_valid_actions_tool= update_valid_actions_tool
        self.generate_candidate_actions_tool = generate_candidate_actions_tool

        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        assert len(self.output_fields) == 1, "ReAct only supports one output field."

        inputs_ = ", ".join([f"`{k}`" for k in self.input_fields.keys()])
        outputs_ = ", ".join([f"`{k}`" for k in self.output_fields.keys()])

        instr = [
            f"You will be given {inputs_} and you will respond with {outputs_}.\n",
            "To do this, you will interleave Thought, Action, and Observation steps.\n",
            "Thought can reason about the current situation, and Action can be the following types:\n",
        ]

        self.tools["Finish"] = dspy.Example(
            name="Finish",
            input_variable=outputs_.strip("`"),
            desc=f"returns the final {outputs_} and finishes the task",
        )

        for idx, tool in enumerate(self.tools):
            tool = self.tools[tool]
            instr.append(
                f"({idx+1}) {tool.name}[{tool.input_variable}], which {tool.desc}"
            )

        instr.append(f"Additionally, every {reflect_interval} steps you will be instructed to reflect on your progress and the strategy taken thus far.")

        instr = "\n".join(instr)
        self.instr = instr

        # self.react = [
        #     Predict(dspy.Signature(self._generate_signature(i), instr))
        #     for i in range(1, max_iters + 1)
        # ]
        self.react = []

    def _generate_signature(self, iters):
        signature_dict = {}
        for key, val in self.input_fields.items():
            signature_dict[key] = val

        for j in range(1, iters + 1):

            # Use self.oracle_insertions to create the Oracle fields
            if j in self.oracle_insertions:
                prefix, _ = self.oracle_insertions[j]
                signature_dict[f"Oracle_{j}"] = dspy.InputField(
                    prefix=f"{prefix}:",
                    format=dsp.passages2text,
                )

            if j % self.reflect_interval == 0:
                signature_dict[f"Reflect_{j}"] = dspy.OutputField(
                    prefix=f"Reflect:",
                    desc=f"self-reflection on your progress and the effectiveness of recent moves",
                )

            if self.read_memory_tool and j > 1:
                signature_dict[f"Memory_{j}"] = dspy.OutputField(
                    prefix=f"Memory:",
                    desc=f"a relevant memory to the current situation",
                    format=dsp.passages2text,
                )
                
            signature_dict[f"Thought_{j}"] = dspy.OutputField(
                prefix=f"Thought:",
                desc="next steps to take based on last observation",
            )

            if self.generate_candidate_actions_tool and j > 1:
                signature_dict[f"CandidateActions_{j}"] = dspy.OutputField(
                    prefix=f"Candidate actions:",
                    desc="a list of valid candidate actions to take based on the last observation and thought",
                )

            tool_list = " or ".join(
                [
                    f"{tool.name}[{tool.input_variable}]"
                    for tool in self.tools.values()
                    if tool.name != "Finish"
                ]
            )
            signature_dict[f"Action_{j}"] = dspy.OutputField(
                prefix=f"Action:",
                desc=f"always either {tool_list} or, when done, Finish[answer]",
            )

            if j < iters:
                signature_dict[f"Observation_{j}"] = dspy.OutputField(
                    prefix=f"Observation:",
                    desc="observations based on action",
                    format=dsp.passages2text,
                )
                
        return signature_dict

    def act(self, output, hop):
        print(f'HOP {hop}')
        try:
            action = output[f"Action_{hop+1}"]
            action_name, action_val = action.strip().split("\n")[0].split("[", 1)
            action_val = action_val.rsplit("]", 1)[0]

            if action_name == "Finish":
                return action_val

            output[f"Observation_{hop+1}"] = self.tools[action_name](action_val)

            if self.write_memory_tool:
                self.write_memory_tool(output[f"Observation_{hop+1}"])

            if self.read_memory_tool and hop > 0:
                output[f"Memory_{hop+2}"] = self.read_memory_tool(output[f"Observation_{hop+1}"]).memory
                
            if self.update_valid_actions_tool and hop > 0:
                self.update_valid_actions_tool(
                    prev_action=action, 
                    observation=output[f"Observation_{hop+1}"])
            
            if self.generate_candidate_actions_tool and hop > 0:
                candidate_actions = self.generate_candidate_actions_tool(
                    # observation=output[f"Observation_{hop+1}"], 
                    thought=output[f"Thought_{hop+1}"]).candidate_actions
                # trim everything after the list of actions
                # candidate_actions = candidate_actions.split("]")[0] + "]"
                output[f"CandidateActions_{hop+1}"] = candidate_actions
            # except AttributeError:
            #     # Handle the case where 'passages' attribute is missing
            #     # TODO: This is a hacky way to handle this. Need to fix this.
            #     output[f"Observation_{hop+1}"] = self.tools[action_name](action_val).passages

        except Exception as e:
            output[f"Observation_{hop+1}"] = (
                "Failed to parse action. Bad formatting or incorrect action name."
            )
            raise e

    def forward(self, **kwargs):
        args = {key: kwargs[key] for key in self.input_fields.keys() if key in kwargs}

        for hop in range(self.max_iters):
            # with dspy.settings.context(show_guidelines=(i <= 2)):

            # We need to do three things for the oracle experiment:
            # (1) Monitor the user's input for oracle injection
            # (2) If input, change the signature to include the oracle (also include in previous signatures)
            # (3) If input, change the args to include the oracle (something like output["Oracle"] = user_input)

            if self.debug:
                for i in range(self.last_printed_output_idx, len(args.keys())):
                    print('\n')
                    print(f"{list(args.keys())[i]}: {list(args.values())[i]}")
                    
            print('\n')
            user_input = input("Insert injection of form 'PREFIX: CONTENT': ")
            print('\n')
            if user_input:
                prefix, content = user_input.split(":")
                prefix = prefix.strip()
                content = content.strip()
                args[f"Oracle_{hop+1}"] = content
                self.oracle_insertions[hop+1] = (prefix, content)
            
            self.react.append(Predict(dspy.Signature(self._generate_signature(hop+1), self.instr)))
            output = self.react[hop](**args)

            if action_val := self.act(output, hop):
                break
            args.update(output)

        # assumes only 1 output field for now - TODO: handling for multiple output fields
        return dspy.Prediction(**{list(self.output_fields.keys())[0]: action_val or ""})
