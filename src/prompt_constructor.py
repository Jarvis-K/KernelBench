import os
from .utils import read_file
import re


"""
Construct Prompt

Design principles: 
- To evaluate base model performance on KernelBench, we use the simplest prompt possible to guide model output to generated desired output format.
- However, we do not do extensive prompt engineering or few-shot example in the LLM to steer behaviour. 
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

def parser_result(result, round=None, wait_flag=False):
    s = f"Here is your Evaluation Result:\n"
    # s = s if not wait_flag else "In this generation, "
    if not result['compiled']:
        s += f"the custom cuda code failed to compile. The error is: {str(result['metadata'])}"
    else:
        s += f"the custom cuda code compiled successfully, "
        if result['correctness']:
            s += f"and the execution result of code is correct. "
            s += f"The average runtime of custom cuda code is {result['runtime']}ms, and the original torch code is {result['baseline_runtime']}ms, the speed up is {result['speed_up']}. "
        else:
            s += f"but the execution result of the code is incorrect. The error is: {str(result['metadata'])[:500]}..."
    return s

def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)


def get_arch_definition(arch_src):
    """
    Construct torch definition from original torch nn.Module definition
    """
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


############################################
# CUDA Prompt
############################################
# PROBLEM_STATEMENT = """You are an expert in CUDA programming and performance optimization. You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
#     You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
# """
PROBLEM_STATEMENT = """You are an expert in CUDA programming and performance optimization. Please write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. You are only limited by your imagination.\n
"""
# PROBLEM_INSTRUCTION = """
# Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
# """
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional on the given GPU device. Just output the new model code and NO testing code!"""

H100_description = """
- GPU Architecture: Hopper
- GPU Memory: 80GB
- Maximum number of registers per thread: 255
- Maximum number of thread blocks per SM: 32
- Shared memory capacity per SM: 228 KB
- Maximum shared memory per thread block: 227 KB
"""
def prompt_generate_custom_cuda(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += PROBLEM_INSTRUCTION
    return prompt

def prompt_generate_custom_cuda_reflection(
    arc_src, example_arch_src, example_new_arch_src, hist_responses, hist_results, recent_hist_flag, best_hist_flag, plan_flag=False, first_step_flag=True, generate_plan_flag=False, plan=None
):
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """

    # prompt += f"""
    # Here's the information of H100 GPU: 
    # {H100_description}
    # """
    if hist_results:
        if best_hist_flag:
            improvement_prompt = f"Below is the previously generated CUDA kernel code:\n"
            for i, (response, result) in enumerate(zip(hist_responses, hist_results)):
                improvement_prompt += f"Round {i+1}:\n"
                improvement_prompt += f"```\n{response}\n```"
                improvement_prompt += f"\n{parser_result(result, round=i+1)}\n\n"
        elif recent_hist_flag:
            result = hist_results[-1]
            response = hist_responses[-1]
            improvement_prompt = f"Below is the most recently generated CUDA kernel code:\n"
            improvement_prompt += f"```\n{response}\n```"
            improvement_prompt += f"\n{parser_result(result)}\n\n"
        else:
            improvement_prompt = f"Below is the previously generated CUDA kernel code:\n"
            for i, (response, result) in enumerate(zip(hist_responses, hist_results)):
                improvement_prompt += f"Round {i+1}:\n"
                improvement_prompt += f"```\n{response}\n```"
                improvement_prompt += f"\n{parser_result(result, round=i+1)}\n\n"

        prompt += improvement_prompt
    
    if not plan_flag:
        # if hist_results:
        #     prompt += "Please generate the best CUDA kernel code for the given architecture based on the previous generations with feedbacks and the differences and optimizations between the previously generated codes."
        prompt += "Please optimize and generate the best CUDA kernel code for the given architecture based on the previous generation with feedbacks. Name your new improved output architecture ModelNew. Output the new code in codeblocks, Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! Please output the complete code in a python code block."
    else:
        if generate_plan_flag:
            if first_step_flag:
                # prompt += "Please generate a short, general and brief generation policy of the custom CUDA kernel for the given architecture. There's no need to generate the code."
                pass
            else:
#                 if "the speed up is" in prompt:
#                     # prompt += "\nGiven the above CUDA codes and corresponding feedbacks, analyze it thoroughly for any potential inefficiencies. Suggest the most important optimization that would significantly improve the code's performance. Provide the clear, actionable recommendation. There's no need to output the complete code, just the optimization directions. Please keep your answers as concise as possible."
#                     prompt += "\nGiven the following CUDA code and the corresponding feedbacks, thoroughly analyze it for possible inefficiencies. Your primary task is to identify the possible inefficient code snippets, fix them and give the optimized ones. There's no need to output the complete code, just the brief, the most important and actionable optimization plan. Your plan should be concise but specific.\nFor each key optimization:\n"
#                 else:
#                     # prompt += "\nGiven the following CUDA code and the corresponding feedbacks, thoroughly analyze it for corresponding errors. Your primary task is to identify the wrong code snippets, fix them and give the correct ones. There's no need to output the complete code, just the brief, the most important and actionable modification. Please keep your answers as concise as possible."
#                     prompt += "\nGiven the following CUDA code and the corresponding feedbacks, thoroughly analyze it for corresponding errors. Your primary task is to identify the wrong code snippets, fix them and give the correct ones. There's no need to output the complete code, just the brief, the most important and actionable modification plan. Your plan should be concise but specific.\nFor each key modification:\n"
#                 prompt += """1. Identify the specific code section to modify
# 2. Describe the exact change needed
# 3. Briefly explain why this change improves performance

# Format your response as:
# 1. One paragraph summary of key performance differences
# 2. Numbered list of specific optimizations, each with:
#     - Target: [Detailed code snippet in the original code]
#     - Change: [specific modification]
#     - Reason: [brief technical explanation]

# Focus only on the most impactful changes."""
                prompt += """Please provide a focused, actionable optimization plan that identifies the MOST CRITICAL differences affecting performance. Your plan should be concise but specific.

For each key optimization (limit to 3-5 most important changes):
1. Identify the specific code section to modify
2. Describe the exact change needed
3. Briefly explain why this change improves performance

Format your response as:
1. One paragraph summary of key performance differences
2. Numbered list of specific optimizations (3-5 items), each with:
   - Target: [Detailed code snippet in the original code]
   - Change: [specific modification]
   - Reason: [brief technical explanation]

Focus only on the most impactful changes.

"""
        else:
            if first_step_flag:
                # prompt += "Here is the generation policy for the given architecture:\n"
                # prompt += plan
                prompt += PROBLEM_INSTRUCTION
            else:
                # prompt += "Here is the optimization directions for the given architecture and the previous generation:\n"
                prompt += plan
                # prompt += "Using the optimization directions or error modification suggestions, rewrite the CUDA code to implement the suggested improvements. Ensure that any errors identified in the original code are fixed, and the optimizations are applied for better performance. Maintain the architecture's original functionality while ensuring it is error-free, runs faster, and is more efficient. Please output the complete code in a code block, not just the CUDA code. Please consider the hardware information of H100 GPU. Just output the new model code and NO testing code! And please Name your optimized output architecture ModelNew."

                prompt += """\nImplement all the changes from the optimization plan. Your implementation must:
1. Follow each optimization step precisely and strictly
2. Maintain functional correctness
3. Name the optimized architecture ModelNew

Important requirements:
- Your implementation must compile and execute correctly
- Focus on implementing the specific optimizations in the plan
- If you encounter any issues with the plan, implement the changes that make technical sense
- Name your optimized architecture ModelNew
- Just output the new model code and NO testing code
- output the complete code in a python code block

Please provide the complete improved code, including all necessary function and class definitions. Add brief comments before each modified section explaining the optimization applied.
"""
            
    return prompt

def prompt_generate_custom_cuda_s1(
    arc_src, example_arch_src, example_new_arch_src, hist_responses, hist_results, wait_responses, wait_results, model
):
    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>" if 'llama' in model or 'Llama' in model else ""
    prompt += PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    if hist_responses:
        improvement_prompt = f"Below is the previously generated CUDA kernel code:\n"
        for i, (response, result) in enumerate(zip(hist_responses, hist_results)):
            improvement_prompt += f"Round {i+1}:\n"
            improvement_prompt += f"```\n{response}\n```"
            improvement_prompt += f"\n{parser_result(result, round=i+1)}\n\n"

        prompt += improvement_prompt

    prompt += """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Here's the code:\n"""
    end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" if "llama" in model or "Llama" in model else "" 
    prompt += end_token
    if wait_responses:
        # prompt += "Below is the CUDA kernel code I've generated and optimized for you:\n"
        for i, response in enumerate(wait_responses):
            prompt += f"{response}\n"
            prompt += "Wait, there are parts of this code that could be improved."
            prompt += f"{parser_result(wait_results[i], round=i+1, wait_flag=True)}\nI'll evaluate the code carefully and regenerate the better one for you. Here's the improved code:\n"

    return prompt


PROBLEM_STATEMENT_CLEANED = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION_CLEANED = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, and NO testing code! NO testing code! NO testing code! \n
"""

def prompt_generate_custom_cuda_fewshot_and_template(ref_arch_src: str, shots: list) -> str:
    """
    Generate a prompt with specified few-shot examples following a template 

    shots: list of few-shot examples to include in the prompt
    Avaliable few shot options to start with: 
    - ex_add: pointwise addition
    - ex_fuse_gelu: fused gelu
    - ex_mnist2: fused convolutions and relus
    - ex_tiled_matmul: tiled matrix multiplication
    """
    prompt = PROBLEM_STATEMENT_CLEANED

    # k = 1
    example_add = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_add.py")
    )
    example_add_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_add.py")
    )
    example_add_desc = "This given architecture is for a pointwise addition: "

    # k = 2
    example_fuse_gelu = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_fuse_gelu.py")
    )
    example_fuse_gelu_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_fuse_gelu.py")
    )
    example_fuse_gelu_desc = "This given architecture is for a fused gelu: "

    # k = 3
    example_mnist2 = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_mnist2.py")
    )
    example_mnist2_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_mnist2.py")
    )
    exmaple_mnist2_desc = "This given architecture is for a model with fused convolutions and relus: "

    # k = 4
    example_tiled_matmul = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_tiled_matmul.py")
    )
    example_tiled_matmul_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_tiled_matmul.py")
    )
    example_tiled_matmul_desc = "This given architecture is for a model with tiled matrix multiplication: "


    examples = []
    for s in shots:
        if s not in ["ex_add", "ex_fuse_gelu", "ex_mnist2", "ex_tiled_matmul"]:
            raise ValueError(f"Invalid shot: {s}")
        elif s == "ex_add":
            examples.append((example_add, example_add_new, example_add_desc))
        elif s == "ex_fuse_gelu":
            examples.append((example_fuse_gelu, example_fuse_gelu_new, example_fuse_gelu_desc))
        elif s == "ex_mnist2":
            examples.append((example_mnist2, example_mnist2_new, exmaple_mnist2_desc))
        elif s == "ex_tiled_matmul":
            examples.append((example_tiled_matmul, example_tiled_matmul_new, example_tiled_matmul_desc))


    for i, tup in enumerate(examples):
        base, kernel, desc = tup

        prompt += f"""
Example {i+1}:\n\n
Here is an example architecture:\n\n
```
{base}
```\n
{PROBLEM_INSTRUCTION_CLEANED} \n
Here is an optimized verison with custom CUDA kernels: \n
```
{kernel}
```\n\n
"""

# should we put task here?
    prompt += f"""
Task:\n\n
Here is an example architecture:\n\n
```
{ref_arch_src}
```\n
"""
    prompt += PROBLEM_INSTRUCTION_CLEANED
    return prompt

def prompt_generate_ex_with_CoT_template(ref_arch_src: str, cot_example: str) -> str:
    """
    Generate a prompt with a CoT example following a template 
    Avaliable CoT examples: 
    - ex_fuse_gelu: fused gelu
    - ex_mnist2: fused convolutions and relus
    - ex_tiled_matmul: tiled matrix multiplication
    """

    # I updated this to allow CoT. Also explicilty state think step by step.
    PROBLEM_INSTRUCTION_COT = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. 
In the end, make sure the final code block contains code for output architecture ModelNew with cuda code.\n
Let's think step by step.\n
""" 

    prompt = PROBLEM_STATEMENT_CLEANED
    
    assert cot_example in ["ex_fuse_gelu", "ex_mnist2", "ex_tiled_matmul"]

    # k = 2
    example_fuse_gelu = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_fuse_gelu.py")
    )
    example_fuse_gelu_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_fuse_gelu.py")
    )
    example_fuse_gelu_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_fuse_gelu.py")
    )
    example_fuse_gelu_desc = "This given architecture is for a fused gelu: "

    # k = 3
    example_mnist2 = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_mnist2.py")
    )
    example_mnist2_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_mnist2.py")
    )
    example_mnist2_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_mnist2.py")
    )
    exmaple_mnist2_desc = "This given architecture is for a model with fused convolutions and relus: "

    # k = 4
    example_tiled_matmul = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_tiled_matmul.py")
    )
    example_tiled_matmul_cot = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_tiled_matmul.py")
    )
    example_tiled_matmul_new = read_file(
        os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_tiled_matmul.py")
    )
    example_tiled_matmul_desc = "This given architecture is for a model with tiled matrix multiplication: "
    
    match cot_example:
        case "ex_fuse_gelu":
            base = example_fuse_gelu
            cot = example_fuse_gelu_cot
            kernel = example_fuse_gelu_new
            desc = example_fuse_gelu_desc
        case "ex_mnist2":
            base = example_mnist2
            cot = example_mnist2_cot
            kernel = example_mnist2_new
            desc = exmaple_mnist2_desc
        case "ex_tiled_matmul":
            base = example_tiled_matmul
            cot = example_tiled_matmul_cot
            kernel = example_tiled_matmul_new
            desc = example_tiled_matmul_desc
        case _:
            raise ValueError(f"Invalid CoT example: {cot_example} not found in CoT examples")

    # construct example with 
    # NOTE: we only do one example with CoT for now
    # 1. ref_src problem -> 2. Instruction -> 3. CoT -> 4. Solution
    prompt += f"""
Here is an example architecture:\n\n
```
{base}
```\n
{PROBLEM_INSTRUCTION_COT} \n
{cot} \n
```
{kernel}
```\n\n
"""

# show task to solve
    prompt += f"""
Task:\n\n
Here is an example architecture:\n\n
```
{ref_arch_src}
```\n
"""
    prompt += PROBLEM_INSTRUCTION_COT

    return prompt



def prompt_generate_custom_cuda_from_file_one_example(ref_arch_src, example_ind=1):
    """
    Deprecated: use prompt_generate_custom_cuda_from_prompt_template instead
    Keep this around for background compatibility
    NOTE: Anne to clean this up
    Check example_ind for prompt templates
    """
    # arch = get_arch_definition_from_file(arch_path)
    arch = ref_arch_src
    # These are strictly defined for now

    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_{example_ind}.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_{example_ind}.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)


def prompt_generate_custom_cuda_from_prompt_template(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)

def prompt_generate_custom_cuda_from_prompt_template_reflection(ref_arch_src: str, hist_responses=None, hist_results=None, recent_hist_flag=False, best_hist_flag=False, example_flag=True, plan_flag=False, first_step_flag=True, generate_plan_flag=False, plan=None) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path) if example_flag else ""
    example_new_arch = read_file(example_new_arch_path) if example_flag else ""

    return prompt_generate_custom_cuda_reflection(arch, example_arch, example_new_arch, hist_responses, hist_results, recent_hist_flag, best_hist_flag, plan_flag, first_step_flag, generate_plan_flag, plan)

def prompt_generate_custom_cuda_from_prompt_template_s1(ref_arch_src: str, hist_responses=None, hist_results=None, example_flag=True, wait_responses=None, wait_results=None, model="llama") -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path) if example_flag else ""
    example_new_arch = read_file(example_new_arch_path) if example_flag else ""

    return prompt_generate_custom_cuda_s1(arch, example_arch, example_new_arch, hist_responses, hist_results, wait_responses, wait_results, model)

def prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src: str, gpu_name: str) -> str:
    """
    Similar to prompt_generate_custom_cuda_from_prompt_template, 
    but with hardware information for the given GPU
    """

    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    gpu_spec_file_path = os.path.join(REPO_TOP_PATH, f"src/prompts/hardware/gpu_specs.py")

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    gpu_spec_info = read_file(gpu_spec_file_path)

    return prompt_generate_prompt_with_hardware_info(
                                        ref_arch_src=arch, 
                                        gpu_name=gpu_name, 
                                        example_arch_src=example_arch, 
                                        example_new_arch_src=example_new_arch, 
                                        gpu_spec_info_src=gpu_spec_info
                                        )
    


def prompt_generate_prompt_with_hardware_info(ref_arch_src: str, 
                                              gpu_name: str, 
                                              example_arch_src: str, 
                                              example_new_arch_src: str, 
                                              gpu_spec_info_src: str) -> str:
    """
    Generate a prompt with hardware information for the given GPU
    gpu_spec_info_src: str of the gpu spec src file
    """

    # Create a dictionary to store the local namespace
    local_dict = {}
    
    # Execute the GPU spec file in the local namespace
    exec(gpu_spec_info_src, {}, local_dict)
    
    # Get the required variables from the local namespace
    GPU_SPEC_INFO = local_dict.get('GPU_SPEC_INFO')
    GPU_DEFINITIONS = local_dict.get('GPU_DEFINITIONS')
    GPU_BEST_PRACTICES = local_dict.get('GPU_BEST_PRACTICES')
    
    if not GPU_SPEC_INFO or not GPU_DEFINITIONS or not GPU_BEST_PRACTICES:
        raise ValueError("GPU_SPEC_INFO or GPU_DEFINITIONS or GPU_BEST_PRACTICES not found in gpu_spec_info_src")

    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """
    
    curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]

    gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
    prompt += f"""
    Here is some information about the underlying hardware that you should keep in mind. \n\n
The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.\n\n"""
    
    for key, value in curr_gpu_spec_info.items():
        if key == "GPU Architecture":
            continue
        prompt += f"""- We have {value} of {key}.\n"""
    
    
    prompt += f"""\n\n
Here are some concepts about the GPU architecture that could be helpful: \n\n"""
    for key, value in GPU_DEFINITIONS.items():
        prompt += f"""- {key}: {value}\n"""

    prompt += f"""\n\n
Here are some best practices for writing CUDA kernels on GPU: \n\n"""
    for best_practice in GPU_BEST_PRACTICES:
        prompt += f"""- {best_practice}\n"""


    prompt += f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """
    

    prompt += PROBLEM_INSTRUCTION
    return prompt


    return Nonoe





def prompt_fix_compile(ref_arch_src, custom_cuda, metadata):
    prompt = PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed to compile:
    ```
    {custom_cuda}
    ```
    Here's the metadata of the compilation error:
    ```
    {metadata}
    ```
    
    Please fix the compilation error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt


def prompt_fix_correctness(ref_arch_src, custom_cuda, metadata):
    prompt = PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed correctness:
    ```
    {custom_cuda}
    ```
    Here's the metadata of the correctness error:
    ```
    {metadata}
    ```
    Please consider how your custom CUDA kernels are implemented, how it is different from the reference implementation, and fix the correctness error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt

def prompt_generate_plan_evaluation(arc_src, kernel_src, plans):
    """
    生成用于评估和整合多个计划的提示
    
    Args:
        plans: 要评估的计划列表
        
    Returns:
        用于评估计划的提示
    """
    prompt = "You are a professional CUDA optimization expert tasked with evaluating multiple CUDA optimization plans and selecting the most effective optimization recommendations."
    prompt += f"""
    You are given the following architecture:
    ```
    {arc_src}
    ```
    You generated the following CUDA kernel code:
    ```
    {kernel_src}
Please carefully analyze the following independently generated CUDA optimization plans and perform the following tasks:

1. Evaluate the effectiveness and feasibility of the optimization recommendations proposed in each plan.
2. Identify the most valuable and potentially performance-enhancing recommendations from each plan.
3. From all the plans, select 3-5 of the most effective optimization recommendations to form a comprehensive optimization plan.
4. Ensure that the selected recommendations do not overlap, are mutually compatible, and can collectively improve performance.

Below are the optimization plans to be evaluated:
"""
    
    for i, plan in enumerate(plans, 1):
        prompt += f"\nPlan {i}:\n```\n{plan}\n```\n"
    
    prompt += """You should output the analysis of all the items in each plan one by one based on the architecture and the kernel code first, and finally, output the final optimization plan in the following format strictly:
<Plan Start>
Optimization Recommendations:
1. Target: [specific code snippet to be modified]
   Change: [specific modification plan]
   Reason: [why this modification can improve performance]

2. Target: [specific code snippet to be modified]
   Change: [specific modification plan]
   Reason: [why this modification can improve performance]

...(3-5 recommendations)
<Plan End>

Please ensure that your final optimization plan is specific, technically feasible, and can significantly improve the performance of the CUDA kernel.
"""
    
    return prompt

def parser_plan_evaluation(response):
    """
    解析计划评估的响应
    
    Args:
        response: 评估响应字符串
        
    Returns:    
        整合后的最终计划
    """
    # 使用正则表达式提取整合后的计划
    plan_pattern = r'<Plan Start>(.*?)<Plan End>'
    match = re.search(plan_pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
    

def main():
    gpu_name = "L40S"


    ref_arch_src = read_file(os.path.join(KERNEL_BENCH_PATH, f"level1/19_ReLU.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"
    prompt = prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src, gpu_name)
    print(prompt)
    # Write prompt to temp file
    temp_file_path = os.path.join(REPO_TOP_PATH, "scratch", "prompt_draft.txt")
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        f.write(prompt)

if __name__ == "__main__":
    main()
