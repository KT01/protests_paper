import pandas as pd
import numpy as np
import re

# read data
acled = pd.read_csv("data/acled_deberta_preds_26_01_2026.csv")

## find unique categories
# make a list of lists len = len(acled)
splitted_list = [actors.split("; ") if isinstance(actors, str) else np.nan for actors in acled["assoc_actor_1"]]

# Flatten the list of lists, checking if each element is iterable (i.e., a list)
flattened_list = [item for sublist in splitted_list for item in (sublist if isinstance(sublist, list) else [sublist])]

# Find unique values, excluding nan
unique_values = {x for x in flattened_list if x == x}

len(unique_values)

# manually select pro-kremlin actors
pro_kremlin = [
    'ER: United Russia',
  'Former Government of Russia (2000-)',
  'Former Military Forces of Russia (2000-)',
  'Former Police Forces of Russia (2000-)',
  'Government of Russia (2000-) Republic of Chechnya',
  'Military Forces of Russia (2000-)',
  'Military Forces of Russia (2000-) National Guard - Special Purpose Police Unit',
  'Motherland Party',
  'NOD: National Liberation Movement',
  'NPSR: National Patriotic Forces of Russia',
  "NPSR: People's Patriotic Union of Russia",
  'Police Forces of Russia (2000-)',
  'VVPOD: Young Army Cadets National Movement',
  "Zakhar Prilepin's Guard"
]

syst_opposition = [
    'CP: A Just Russia',
    'KPRF: Communist Party of the Russian Federation',
    'KR: Communists of Russia',
    'LDPR: Liberal Democratic Party of Russia',
    'New People',
    'PR: Party of Growth',
    'Yabloko: Russian United Democratic Party'

]
# Function to check if any sub-value is in pro_kremlin
def check_pro_kremlin(value):
    """
    input: str
    output: int {0, 1, 2}

    example: 'Government of Russia (2000-); Labour Group (Russia)' ->
              ['Government of Russia (2000-)', 'Labour Group (Russia)'] ->
                1
    """
    if pd.isna(value):
        return pd.NA

    actors = str(value).split('; ')

    if any(actor in pro_kremlin for actor in actors):
        return "pro_kremlin"
    elif any(actor in syst_opposition for actor in actors):
        return "syst_opposition"
    else:
        return "other"

# Apply the function to each row in the 'assoc_actor_1' column
acled['org_indicator'] = acled['assoc_actor_1'].apply(check_pro_kremlin)

# write csv file
acled.to_csv("data/acled_with_org_indicator.csv", index=False)