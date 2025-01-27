import pandas as pd

def configure_weather(weather_file):
    """
    Load and configure the weather dataset by mapping meeting_key to raceId.s
    
    Parameters:
        weather_file (str): Path to the weather dataset CSV file.
        mapping_file (str): Path to the mapping CSV file (meeting_key -> raceId).
        
    Returns:
        pd.DataFrame: Weather dataset with raceId included.
    """

    mapping = {
    1140: 1053,
    1141: 1054,
    1142: 1055,
    1143: 1056,
    1144: 1057,
    1145: 1058,
    1146: 1059,
    1147: 1060,
    1148: 1061,
    1149: 1062, 
    1150: 1063,
    1151: 1064,
    1152: 1065,
    1153: 1066,
    1154: 1067,
    1155: 1068,
    1156: 1069,
    1157: 1070,
    1158: 1071,
    1159: 1072,
    1160: 1073,
    1161: 1074,
    1162: 1075,
    1163: 1076,
    1164: 1077,
    1165: 1078,
    1166: 1079,
    1167: 1080,
    1168: 1081,
    1169: 1082,
    1170: 1083,
    1171: 1084,
    1172: 1085,
    1173: 1086,
    1174: 1087,
    1175: 1088,
    1176: 1089,
    1177: 1090,
    1178: 1091,
    1179: 1092,
    1180: 1093,
    1181: 1094,
    1182: 1095,
    1183: 1096,
    1184: 1097,
    1185: 1098,
    1186: 1099,
    1187: 1100,
    1188: 1101,
    1189: 1102,
    1190: 1103,
    1191: 1104,
    1192: 1105,
    1193: 1106,
    1194: 1107,
    1195: 1108,
    1196: 1109,
    1197: 1110,
    1198: 1111,
    1199: 1112,
    1200: 1113,
    1201: 1114,
    1202: 1115,
    1203: 1116,
    1204: 1117,
    1205: 1118,
    1206: 1119,
    1207: 1120,
    1208: 1121,
    1209: 1122,
    1210: 1123,
    1211: 1124,
    1212: 1125,
    1213: 1126,
    1214: 1127,
    1215: 1128,
    1216: 1129,
    1217: 1130,
    1218: 1131,
    1219: 1132,
  }
    
   # Convert the mapping dictionary into a DataFrame
    mapping_df = pd.DataFrame(list(mapping.items()), columns=["meeting_key", "raceId"])

    # Load the weather dataset
    weather_df = pd.read_csv(weather_file)

    # Merge datasets
    weather_with_raceId = pd.merge(weather_df, mapping_df, on="meeting_key", how="left")
    weather_with_raceId = weather_with_raceId.dropna()
    return weather_with_raceId

# Specify the path to your weather dataset CSV file
weather_file_path = r"C:\Users\Albin Binu\Documents\College\Year 4\Final Year Project\f1_project_env\data\meetings_weather.csv"

# Call the function and print the result
weather_with_raceId = configure_weather(weather_file_path)
print(weather_with_raceId.to_string)