from typing import List


def general(
    metabolic_delay:int,
    metabolic_damage:int,
    object_variables:str,
    array_variables:str,
    actor_functions:str,
    spawn_relative:bool = True,
    spawn_range:float = 1000.0
):
    return """\
// Directives
#import "acs/retinal.acs"
#include "zcommon.acs"

script "Load Config Information" OPEN {{

    // Metabolic variables
    metabolic_delay = {metabolic_delay};
    metabolic_damage = {metabolic_damage};

    // Arena settings
    // Spawn behaviour
    spawn_relative = {spawn_relative};
    spawn_range = {spawn_range};

    // Object Variables
    {object_variables}

    // Loading arrays
    {array_variables}
}}

// Actor Functions / Scripts
{actor_functions}""".format(
        metabolic_delay=metabolic_delay,
        metabolic_damage=metabolic_damage,
        object_variables=object_variables,
        array_variables=array_variables,
        actor_functions=actor_functions,
        spawn_relative=str(spawn_relative).lower(),
        spawn_range=spawn_range
    )


def object_variables(typ:str, unique:int, init:int, delay:int):
    return """
    // {type} variables
    {type}_unique = {unique};
    {type}_init = {init};
    {type}_delay = {delay};
""".format(
        type=typ, unique=unique, init=init, delay=delay
    )


def actor_function(actor_name:str, values: List[int], heal_or_damage:bool):
    actor_name = actor_name.replace("-", "_")
    # Actor name will be used in function name, not possible with -
    num_values = len(values)
    values_string = ",".join([str(v) for v in values])

    return """\
int values_{actor_name}[{num_values}] = {{ {values} }};
script "func_{actor_name}" (void)
{{
    int i = Random(0,{num_values}-1);
    {heal_or_damage}Thing(values_{actor_name}[i]);
}}
""".format(
        actor_name=actor_name,
        values=values_string,
        num_values=num_values,
        heal_or_damage=heal_or_damage,
    )


def heal_function(actor_name:str, values: List[int]):
    return actor_function(actor_name, values, "Heal")


def damage_function(actor_name:str, values: List[int]):
    return actor_function(actor_name, values, "Damage")


def actor_arrays(index:int, actor_name:str, num_textures:int):
    return """
    actor_names[{index}] = "{actor_name}";
    actor_num_textures[{index}] = {num_textures};
""".format(
        index=index, actor_name=actor_name, num_textures=num_textures
    )
