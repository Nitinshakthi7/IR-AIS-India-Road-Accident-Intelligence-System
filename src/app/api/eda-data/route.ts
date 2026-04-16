import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const edaData = JSON.parse(
      fs.readFileSync(
        path.join(process.cwd(), "ml-service/models/eda_data.json"),
        "utf-8"
      )
    );

    // Transform severity_distribution to array format
    const severityColors: Record<string, string> = {
      "Slight Injury": "#22c55e",
      "Serious Injury": "#f59e0b",
      "Fatal injury": "#ef4444",
    };
    const severityDistribution = Object.entries(
      (edaData.severity_distribution || {}) as Record<string, number>
    ).map(([name, value]) => ({
      name,
      value,
      color: severityColors[name] || "#6b7280",
    }));

    // Transform hour_distribution to sorted array
    const peakHours = Object.entries(
      (edaData.hour_distribution || {}) as Record<string, number>
    )
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([hour, count]) => ({
        hour: `${hour.padStart(2, "0")}:00`,
        count,
      }));

    // Transform day_distribution to array with proper day order
    const dayOrder = [
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday",
    ];
    const dayDistribution = dayOrder.map((day) => ({
      day,
      count: ((edaData.day_distribution || {}) as Record<string, number>)[day] || 0,
    }));

    // Transform weather_severity_cross to flat array for heatmap
    const weatherSeverity: {
      weather: string;
      severity: string;
      count: number;
    }[] = [];
    for (const [weather, severities] of Object.entries(
      (edaData.weather_severity_cross || {}) as Record<
        string,
        Record<string, number>
      >
    )) {
      for (const [severity, count] of Object.entries(severities)) {
        weatherSeverity.push({ weather, severity, count });
      }
    }

    // Transform age_severity_cross to flat array
    const ageSeverity: { age: string; severity: string; count: number }[] = [];
    for (const [age, severities] of Object.entries(
      (edaData.age_severity_cross || {}) as Record<string, Record<string, number>>
    )) {
      for (const [severity, count] of Object.entries(severities)) {
        ageSeverity.push({ age, severity, count });
      }
    }

    // Vehicle type distribution
    const vehicleTypeDist = Object.entries(
      (edaData.vehicle_type_distribution || {}) as Record<string, number>
    ).map(([name, count]) => ({ name, count }));

    // Cause distribution
    const causeDist = Object.entries(
      (edaData.cause_distribution || {}) as Record<string, number>
    ).map(([name, count]) => ({ name, count }));

    const pedestrianDist = Object.entries(
      (edaData.pedestrian_distribution || {}) as Record<string, number>
    ).map(([name, value]) => ({ name, value }));

    const collisionDist = Object.entries(
      (edaData.collision_distribution || {}) as Record<string, number>
    ).map(([name, count]) => ({ name, count }));

    const defectDist = Object.entries(
      (edaData.defect_distribution || {}) as Record<string, number>
    ).map(([name, count]) => ({ name, count }));

    const areaDist = Object.entries(
      (edaData.area_distribution || {}) as Record<string, number>
    ).map(([name, count]) => ({ name, count }));

    const casualtyDist = Object.entries(
      (edaData.casualty_distribution || {}) as Record<string, number>
    ).map(([count, occurrences]) => ({ count, occurrences }));

    return NextResponse.json({
      severity_distribution: severityDistribution,
      peak_hours: peakHours,
      day_distribution: dayDistribution,
      weather_severity: weatherSeverity,
      age_severity: ageSeverity,
      vehicle_type_distribution: vehicleTypeDist,
      cause_distribution: causeDist,
      pedestrian_distribution: pedestrianDist,
      collision_distribution: collisionDist,
      defect_distribution: defectDist,
      area_distribution: areaDist,
      casualty_distribution: casualtyDist,
      total_records: edaData.total_records,
      total_features: edaData.feature_count,
    });
  } catch (error) {
    console.error("EDA data error:", error);
    return NextResponse.json(
      { error: "Failed to load EDA data" },
      { status: 500 }
    );
  }
}
