export interface FeatureOption {
  label: string;
  options: string[];
  type?: "select" | "number";
  min?: number;
  max?: number;
  placeholder?: string;
}

export const Day_of_week: FeatureOption = {
  label: "Day of Week",
  options: [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ],
};

export const Age_band_of_driver: FeatureOption = {
  label: "Age Band of Driver",
  options: ["18-30", "31-50", "Under 18", "Over 51", "Unknown"],
};

export const Sex_of_driver: FeatureOption = {
  label: "Sex of Driver",
  options: ["Male", "Female", "Unknown"],
};

export const Educational_level: FeatureOption = {
  label: "Educational Level",
  options: [
    "Above high school",
    "Junior high school",
    "Elementary school",
    "High school",
    "Writing & reading",
    "Unknown",
  ],
};

export const Vehicle_driver_relation: FeatureOption = {
  label: "Vehicle Driver Relation",
  options: ["Employee", "Owner", "Other", "Unknown"],
};

export const Driving_experience: FeatureOption = {
  label: "Driving Experience",
  options: [
    "1-2yr",
    "Above 10yr",
    "5-10yr",
    "2-5yr",
    "Below 1yr",
    "No Licence",
    "Unknown",
  ],
};

export const Type_of_vehicle: FeatureOption = {
  label: "Type of Vehicle",
  options: [
    "Automobile",
    "Public (> 45 seats)",
    "Lorry (41?100Q)",
    "Public (13?45 seats)",
    "Long lorry",
    "Taxi",
    "Pick up upto 10Q",
    "Stationwagen",
    "Ridden horse",
    "Special vehicle",
    "Bajaj",
    "Motorcycle",
    "Turbo",
    "Other",
    "Bicycle",
    " horse",
  ],
};

export const Owner_of_vehicle: FeatureOption = {
  label: "Owner of Vehicle",
  options: ["Owner", "Governmental", "Organization", "Other"],
};

export const Service_year_of_vehicle: FeatureOption = {
  label: "Service Year of Vehicle",
  options: [
    "Above 10yr",
    "5-10yrs",
    "1-2yr",
    "2-5yrs",
    "Below 1yr",
    "Unknown",
  ],
};

export const Defect_of_vehicle: FeatureOption = {
  label: "Defect of Vehicle",
  options: ["No defect", "Defective"],
};

export const Area_accident_occured: FeatureOption = {
  label: "Area Accident Occurred",
  options: [
    "Residential areas",
    "Office areas",
    "Recreational areas",
    "Industrial areas",
    "Other",
    "Church areas",
    "School areas",
    "Rural village areas",
    "Market areas",
    "Hospital areas",
    "Rural village areasOffice areas",
    "Unknown",
    "Bridge",
  ],
};

export const Lanes_or_Medians: FeatureOption = {
  label: "Lanes or Medians",
  options: [
    "Undivided Two way",
    "Two-way (divided with broken lines road marking)",
    "One way",
    "other",
    "Two-way (divided with solid lines road marking)",
  ],
};

export const Road_allignment: FeatureOption = {
  label: "Road Alignment",
  options: [
    "Tangent road with flat terrain",
    "Tangent road with mild grade and flat terrain",
    "Tangent road with steep grade and flat terrain",
    "Lanes or Medians",
    "other",
    "Tangent road with mild grade and mountainous terrain",
    "Tangent road with steep grade and mountainous terrain",
    "Escarpments",
  ],
};

export const Types_of_Junction: FeatureOption = {
  label: "Types of Junction",
  options: [
    "No junction",
    "Y Shape",
    "Crossing",
    "O Shape",
    "T Shape",
    "X Shape",
    "Others",
  ],
};

export const Road_surface_type: FeatureOption = {
  label: "Road Surface Type",
  options: ["Asphalt roads", "Earth roads", "Gravel roads", "Other"],
};

export const Road_surface_conditions: FeatureOption = {
  label: "Road Surface Conditions",
  options: ["Dry", "Wet or damp", "Snow", "Flood over 3cm. depth"],
};

export const Light_conditions: FeatureOption = {
  label: "Light Conditions",
  options: [
    "Daylight",
    "Darkness - lights lit",
    "Darkness - no lighting",
    "Darkness - lights unlit",
  ],
};

export const Weather_conditions: FeatureOption = {
  label: "Weather Conditions",
  options: [
    "Normal",
    "Raining",
    "Raining and Snowing",
    "Cloudy",
    "Other",
    "Windy",
    "Fog or mist",
    "Snow",
  ],
};

export const Type_of_collision: FeatureOption = {
  label: "Type of Collision",
  options: [
    "Collision with roadside-parked vehicles",
    "Vehicle with vehicle collision",
    "Collision with roadside objects",
    "Collision with animals",
    "Other",
    "Rollover",
    "Fall from vehicles",
    "Collision with pedestrians",
    "With tram",
  ],
};

export const Vehicle_movement: FeatureOption = {
  label: "Vehicle Movement",
  options: [
    "Going straight",
    "Overtaking",
    "Changing lane to the left",
    "Changing lane to the right",
    "U-Turn",
    "Turning right",
    "Turning left",
    "Reversing",
    "Stopping",
    "Parked",
    "Waiting to go",
    "Entering from left",
    "Entering from right",
    "Moving Backward",
    "Negotiating a curve",
    "unknown",
  ],
};

export const Pedestrian_movement: FeatureOption = {
  label: "Pedestrian Movement",
  options: [
    "Not a Pedestrian",
    "Crossing from nearside",
    "Crossing from offside",
    "Standing in roadway",
    "Walking along roadway",
    "Injured in vehicle",
    "Unknown or other",
  ],
};

export const Cause_of_accident: FeatureOption = {
  label: "Cause of Accident",
  options: [
    "Moving Backward",
    "Overtaking",
    "Changing lane to the left",
    "Changing direction to the right",
    "Changing direction to the left",
    "Avoidance maneuvers",
    "Driving at high speed",
    "Driving to the left",
    "Driving under the influence of drugs",
    "No distancing",
    "Getting off the vehicle improperly",
    "Driving at unsafe speed",
    "Improper parking",
    "Overloading",
    "No priority to pedestrian",
    "No priority to vehicle",
    "Not keeping proper distance",
    "Priority to the right",
    "Ignoring traffic signals",
    "Improper driving maneuvers",
    "Unknown",
    "Improper lane change",
    "Overspeed",
    "Turn left/right without caution",
    "Driverless vehicle",
    "Following too close",
    "Other",
    "Passed at a red traffic light",
    "Moving straight",
    "Distraction",
    "Lane changing",
    "changing lane",
  ],
};

export const Number_of_vehicles_involved: FeatureOption = {
  label: "Number of Vehicles Involved",
  options: [],
  type: "number",
  min: 1,
  max: 10,
  placeholder: "e.g. 2",
};

export const Hour_of_Day: FeatureOption = {
  label: "Hour of Day (0–23)",
  options: [],
  type: "number",
  min: 0,
  max: 23,
  placeholder: "e.g. 17",
};

/** Map of all feature names to their option definitions */
export const ALL_FEATURE_OPTIONS: Record<string, FeatureOption> = {
  Day_of_week,
  Age_band_of_driver,
  Sex_of_driver,
  Educational_level,
  Vehicle_driver_relation,
  Driving_experience,
  Type_of_vehicle,
  Owner_of_vehicle,
  Service_year_of_vehicle,
  Defect_of_vehicle,
  Area_accident_occured,
  Lanes_or_Medians,
  Road_allignment,
  Types_of_Junction,
  Road_surface_type,
  Road_surface_conditions,
  Light_conditions,
  Weather_conditions,
  Type_of_collision,
  Number_of_vehicles_involved,
  Vehicle_movement,
  Pedestrian_movement,
  Cause_of_accident,
  Hour_of_Day,
};

/** Ordered list of all feature field names used for prediction input */
export const ALL_FEATURES: string[] = [
  "Day_of_week",
  "Age_band_of_driver",
  "Sex_of_driver",
  "Educational_level",
  "Vehicle_driver_relation",
  "Driving_experience",
  "Type_of_vehicle",
  "Owner_of_vehicle",
  "Service_year_of_vehicle",
  "Defect_of_vehicle",
  "Area_accident_occured",
  "Lanes_or_Medians",
  "Road_allignment",
  "Types_of_Junction",
  "Road_surface_type",
  "Road_surface_conditions",
  "Light_conditions",
  "Weather_conditions",
  "Type_of_collision",
  "Number_of_vehicles_involved",
  "Vehicle_movement",
  "Pedestrian_movement",
  "Cause_of_accident",
  "Hour_of_Day",
];
