import inspect
from pprint import pprint

from importlib import import_module

# Map methods to their classes
METHOD_CLASSES = {
    'rtts': ('rtts', 'RTTS'),
    'mlmp': ('mlmp', 'MLMP'),
    'clipartt': ('clipartt', 'CLIPARTT'),
    'watt': ('watt', 'WATT'),
    'tpt': ('tpt', 'TPT'),
    'tent': ('tent', 'TENT'),
}



# Main function
def get_method(args, device):
    if args.method not in METHOD_CLASSES:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Retrieve the class and its arguments
    module_name, class_name = METHOD_CLASSES[args.method]
    method_class = getattr(import_module(f'.{module_name}', __package__), class_name)
    class_args = get_class_args(method_class)
    
    # Filter relevant args and add additional variables
    args_dict = vars(args)
    method_args = {key: args_dict[key] for key in class_args if key in args_dict}
    method_args['device'] = device  # Add device explicitly if needed

    # Check for missing arguments
    signature = inspect.signature(method_class.__init__)
    missing_args = [key for key in class_args if key not in method_args
                    and signature.parameters[key].default is inspect.Parameter.empty]
    if missing_args:
        raise ValueError(f"Missing arguments for {method_class.__name__}: {missing_args}")
    
    # Summarize arguments for better printing
    # summarized_args = summarize_args(method_args)

    # Log selected method and parameters in pretty form
    # print("\nMethod +++++++++++++++++++++++++++++++++++++")
    #
    # print(f"Selected Method: {method_class.__name__}")
    # print("Method Parameters:")
    # pprint(summarized_args)
    # print("----------------------------------------")

    # Instantiate the class with relevant arguments
    method = method_class(**method_args)
    if args.method != 'rtts' and getattr(args, 'with_rtts', False) and args.method != 'tpt':
        from ovss import configure_refinement
        configure_refinement(method.model)
    return method


# Function to get class arguments dynamically
def get_class_args(cls):
    signature = inspect.signature(cls.__init__)
    return [param.name for param in signature.parameters.values() if param.name != 'self']

# Function to summarize the 'classes' argument
def summarize_args(method_args):
    summarized_args = method_args.copy()
    if 'classes' in summarized_args:
        summarized_args['classes'] = len(summarized_args['classes'])  # Replace with total count
    return summarized_args
