import pandas as pd

df = pd.read_csv("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

# Check requires_road_closure distribution across status
print("requires_road_closure vs status:")
print(pd.crosstab(df['status'], df['requires_road_closure']))

# Check assigned_to_police_id vs requires_road_closure
print("\nassigned_to_police_id non-null vs requires_road_closure:")
print(pd.crosstab(df['assigned_to_police_id'].notnull(), df['requires_road_closure']))

# Check citizen_accident_id vs requires_road_closure
print("\ncitizen_accident_id non-null vs requires_road_closure:")
print(pd.crosstab(df['citizen_accident_id'].notnull(), df['requires_road_closure']))

# Check client_id vs requires_road_closure
print("\nclient_id vs requires_road_closure:")
print(pd.crosstab(df['client_id'], df['requires_road_closure']))

# Check route_path vs requires_road_closure
print("\nroute_path non-null vs requires_road_closure:")
print(pd.crosstab(df['route_path'].notnull(), df['requires_road_closure']))
