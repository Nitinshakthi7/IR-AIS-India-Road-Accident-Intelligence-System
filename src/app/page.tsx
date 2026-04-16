"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect, useCallback, Suspense } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  AreaChart,
  Area,
  Label as RechartsLabel,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/frontend/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/frontend/components/ui/select";
import { Button } from "@/frontend/components/ui/button";
import { Label } from "@/frontend/components/ui/label";
import { Badge } from "@/frontend/components/ui/badge";
import { Separator } from "@/frontend/components/ui/separator";
import { Skeleton } from "@/frontend/components/ui/skeleton";
import { Alert, AlertDescription } from "@/frontend/components/ui/alert";
import { Progress } from "@/frontend/components/ui/progress";
import { Input } from "@/frontend/components/ui/input";
import { cn } from "@/frontend/lib/utils";
import {
  BarChart3,
  BrainCircuit,
  Zap,
  Database,
  Layers,
  Activity,
  TrendingUp,
  Shield,
  ShieldAlert,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Thermometer,
  CloudRain,
  Car,
  Eye,
  Loader2,
  Info,
  Star,
  Trophy,
  Target,
  User,
  Truck,
  MapPin,
  Wrench,
  ChevronDown,
  ChevronUp,
  PieChart as PieChartIcon,
  Flame,
  Calendar,
  Route,
} from "lucide-react";
import { ALL_FEATURE_OPTIONS, ALL_FEATURES } from "@/frontend/lib/feature-options";
import { LandingHero } from "@/frontend/components/LandingHero";
import { CustomPathway } from "@/frontend/components/PathwayLine";
import { NarrativeJourney } from "@/frontend/components/NarrativeJourney";
import { FloatingSafeRoadsCoin } from "@/frontend/components/FloatingSafeRoadsCoin";
import { FooterIllustration } from "@/frontend/components/FooterIllustration";
import Image from "next/image";

// ─── Types ──────────────────────────────────────────────────────────

interface SeverityItem {
  name: string;
  value: number;
  color: string;
}

interface PeakHour {
  hour: string;
  count: number;
}

interface DayItem {
  day: string;
  count: number;
}

interface HeatmapItem {
  weather: string;
  severity: string;
  count: number;
}

interface AgeItem {
  age: string;
  severity: string;
  count: number;
}

interface GenericDistributionItem {
  name: string;
  count: number;
}

interface CasualtyItem {
  count: string;
  occurrences: number;
}

interface EdaData {
  severity_distribution: SeverityItem[];
  peak_hours: PeakHour[];
  day_distribution: DayItem[];
  weather_severity: HeatmapItem[];
  age_severity: AgeItem[];
  vehicle_type_distribution: GenericDistributionItem[];
  cause_distribution: GenericDistributionItem[];
  pedestrian_distribution: { name: string, value: number }[];
  collision_distribution: GenericDistributionItem[];
  defect_distribution: GenericDistributionItem[];
  area_distribution: GenericDistributionItem[];
  casualty_distribution: CasualtyItem[];
  total_records: number;
  total_features: number;
  error?: string;
}

interface ClassModel {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  approach: string;
  rocAuc: number | null;
  confusionMatrix: number[][];
  bestParams: Record<string, unknown> | null;
}

interface RegModel {
  name: string;
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  approach: string;
  bestParams: Record<string, unknown> | null;
}

interface ModelData {
  classification: ClassModel[];
  regression: RegModel[];
  auxiliary?: Record<string, Record<string, any>>;
}

// ─── Chart Theme ────────────────────────────────────────────────────

const tooltipStyle: React.CSSProperties = {
  backgroundColor: "#ffffff",
  border: "2px solid #000000",
  borderRadius: "12px",
  color: "#1a1a1a",
  boxShadow: "4px 4px 0px 0px rgba(0,0,0,1)",
  padding: "8px 12px",
};

const axisTickStyle = { fill: "#4b4b4b", fontSize: 11, fontWeight: 500 };

// ─── Helper Components ──────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, color, gradient }: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  color: string;
  gradient: string;
}) {
  return (
    <div className={cn("fp-card fp-card-hover group relative overflow-hidden", gradient)}>
      {/* Decorative line art background */}
      <div className="absolute top-0 right-0 p-2 opacity-[0.05] group-hover:opacity-10 transition-opacity">
        <Icon className="h-24 w-24 -mr-8 -mt-8 rotate-12" />
      </div>
      <div className="flex items-center gap-5 relative z-10">
        <div className={cn(
          "flex h-14 w-14 items-center justify-center rounded-2xl border-2 border-black shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] transition-transform duration-300 group-hover:-translate-y-1",
          color === "text-blue-400" && "bg-blue-100 text-blue-700",
          color === "text-emerald-400" && "bg-emerald-100 text-emerald-700",
          color === "text-red-400" && "bg-red-100 text-red-700",
          color === "text-amber-400" && "bg-amber-100 text-amber-700",
        )}>
          <Icon className="h-7 w-7" />
        </div>
        <div>
          <p className="text-xs font-bold uppercase tracking-widest text-black/60 mb-1">{label}</p>
          <p className={cn("text-4xl font-serif font-bold tracking-tight text-black animate-count-up")}>
            {value}
          </p>
        </div>
      </div>
    </div>
  );
}

function ChartCard({ title, description, icon: Icon, accentColor, children, className }: {
  title: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  accentColor?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("fp-card fp-card-hover min-h-full flex flex-col", className)}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          {Icon && (
            <div className={cn("flex h-10 w-10 items-center justify-center rounded-xl border-2 border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]",
              accentColor === "amber" && "bg-amber-100 text-amber-700",
              accentColor === "blue" && "bg-blue-100 text-blue-700",
              accentColor === "violet" && "bg-purple-100 text-purple-700",
              accentColor === "cyan" && "bg-cyan-100 text-cyan-700",
              accentColor === "red" && "bg-red-100 text-red-700",
              accentColor === "emerald" && "bg-emerald-100 text-emerald-700",
              accentColor === "rose" && "bg-rose-100 text-rose-700",
              !accentColor && "bg-secondary text-black",
            )}>
              <Icon className="h-5 w-5" />
            </div>
          )}
          <div>
            <h3 className="text-lg font-serif font-bold text-black leading-tight">{title}</h3>
            {description && <p className="text-xs font-medium text-black/50 mt-0.5">{description}</p>}
          </div>
        </div>
      </div>
      <div className="flex-1 min-h-0">{children}</div>
    </div>
  );
}

// ─── Confidence Gauge ───────────────────────────────────────────────

function ConfidenceGauge({ value, label, color }: { value: number; label: string; color: string }) {
  const circumference = 2 * Math.PI * 40;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center gap-2">
      <svg width="120" height="120" className="-rotate-90">
        <circle cx="60" cy="60" r="40" fill="none" stroke="#ebdec5" strokeWidth="10" />
        <circle cx="60" cy="60" r="40" fill="none" className={color} strokeWidth="10" strokeLinecap="round"
          stroke="currentColor"
          strokeDasharray={circumference} strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1)' }} />
      </svg>
      <div className="absolute flex flex-col items-center justify-center inset-0">
        <span className="text-2xl font-serif font-bold">{value.toFixed(0)}%</span>
        <span className="text-[10px] font-bold uppercase tracking-wider text-black/60">{label}</span>
      </div>
    </div>
  );
}

// ─── Weather Heatmap ───────────────────────────────────────────────

function WeatherHeatmapChart({ data }: { data: HeatmapItem[] }) {
  const weathers = [...new Set(data.map((d) => d.weather))];
  const severities = ["Fatal injury", "Serious Injury", "Slight Injury"];
  const maxCount = Math.max(...data.map((d) => d.count), 1);

  const getColor = (count: number) => {
    const ratio = count / maxCount;
    if (ratio > 0.7) return "bg-[#e05a47] text-white";
    if (ratio > 0.4) return "bg-[#d4a843] text-black";
    if (ratio > 0.15) return "bg-[#6bc4b3] text-black";
    return "bg-[#ebdec5] text-black/70";
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-separate border-spacing-y-2">
        <thead>
          <tr>
            <th className="text-left py-2 px-3 text-black/40 font-bold text-[10px] uppercase tracking-widest">
              Weather
            </th>
            {severities.map((s) => (
              <th
                key={s}
                className="text-center py-2 px-3 text-black/40 font-bold text-[10px] uppercase tracking-widest"
              >
                {s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {weathers.map((w) => (
            <tr key={w} className="group transition-colors">
              <td className="py-2 px-3 text-sm font-bold text-black/80">{w}</td>
              {severities.map((s) => {
                const item = data.find(
                  (d) => d.weather === w && d.severity === s
                );
                const count = item?.count || 0;
                return (
                  <td key={s} className="py-1 px-1">
                    <div
                      className={cn(
                        "rounded-xl px-3 py-2 text-center text-xs font-bold border-2 border-black/5 transition-all hover:scale-105 hover:border-black",
                        getColor(count)
                      )}
                    >
                      {count.toLocaleString()}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Age Severity Chart ────────────────────────────────────────────

function AgeSeverityChart({ data }: { data: AgeItem[] }) {
  const ages = [...new Set(data.map((d) => d.age))];
  const severities = ["Fatal injury", "Serious Injury", "Slight Injury"];
  const severityColors: Record<string, string> = {
    "Fatal injury": "#e05a47",
    "Serious Injury": "#d4a843",
    "Slight Injury": "#6bc4b3",
  };

  const chartData = ages.map((age) => {
    const row: Record<string, string | number> = { age };
    for (const s of severities) {
      const item = data.find((d) => d.age === age && d.severity === s);
      row[s] = item?.count || 0;
    }
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" vertical={false} />
        <XAxis dataKey="age" tick={axisTickStyle} axisLine={{ stroke: '#000', strokeWidth: 1 }} />
        <YAxis tick={axisTickStyle} axisLine={{ stroke: '#000', strokeWidth: 1 }} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
        <Legend 
          wrapperStyle={{ paddingTop: 20, fontWeight: 700, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }} 
          iconType="circle"
        />
        {severities.map((s) => (
          <Bar
            key={s}
            dataKey={s}
            fill={severityColors[s]}
            radius={[4, 4, 0, 0]}
            stackId="a"
            stroke="#000"
            strokeWidth={1.5}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Dashboard Skeleton ────────────────────────────────────────────

function DashboardSkeleton() {
  return (
    <div className="space-y-8">
      {/* Hero skeleton */}
      <div className="h-48 w-full rounded-2xl border-2 border-black/10 bg-secondary/20" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-32 rounded-2xl border-2 border-black/10 bg-secondary/10" />
        ))}
      </div>
    </div>
  );
}

// ─── Model Skeleton ────────────────────────────────────────────────

function ModelSkeleton() {
  return (
    <div className="space-y-8">
      <div className="flex justify-center gap-4">
        <div className="h-12 w-48 rounded-full border-2 border-black/10 bg-secondary/20" />
        <div className="h-12 w-48 rounded-full border-2 border-black/10 bg-secondary/20" />
      </div>
      {[1, 2].map((i) => (
        <div key={i} className="h-64 rounded-2xl border-2 border-black/10 bg-secondary/10" />
      ))}
    </div>
  );
}

// ─── Predictor Skeleton ────────────────────────────────────────────

function PredictorSkeleton() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="h-[600px] rounded-2xl border-2 border-black/10 bg-secondary/10" />
      <div className="h-[600px] rounded-2xl border-2 border-black/10 bg-secondary/10 border-dashed" />
    </div>
  );
}

// ─── 1. DASHBOARD TAB ─────────────────────────────────────────────

function DashboardTab() {
  const [edaData, setEdaData] = useState<EdaData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/eda-data")
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          console.error("EDA API Error:", data.error);
          setEdaData(null);
        } else {
          setEdaData(data);
        }
        setLoading(false);
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  }, []);

  if (loading) return <DashboardSkeleton />;
  if (!edaData)
    return (
      <div className="fp-card border-destructive text-destructive font-bold text-center py-10">
        Failed to load EDA data. Make sure the ML service data files exist.
      </div>
    );

  const fatalCount =
    edaData.severity_distribution.find((s) => s.name === "Fatal injury")
      ?.value || 0;
  const seriousCount =
    edaData.severity_distribution.find((s) => s.name === "Serious Injury")
      ?.value || 0;

  return (
    <div className="space-y-10 pb-20">
      {/* Hero Banner - Redesigned */}
      <section className="relative overflow-hidden rounded-3xl border-[3px] border-black bg-white p-8 md:p-12 shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] animate-fade-in-up">
        {/* Decorative elements */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/20 rounded-full -mr-32 -mt-32 blur-3xl" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-secondary/40 rounded-full -ml-24 -mb-24 blur-3xl" />
        
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-12">
          <div className="max-w-2xl">
            <div className="flex items-center gap-3 text-black font-bold uppercase tracking-[0.2em] text-xs mb-6">
              <Activity className="h-4 w-4 text-primary" />
              <span>Exploratory Data Analysis</span>
              <div className="h-px w-12 bg-black/10" />
            </div>
            <h2 className="editorial-title mb-6 leading-[1.1]">
              Deciphering the <br/>
              <span className="text-primary italic">Geometry</span> of Risk.
            </h2>
            <p className="text-lg font-medium text-black/60 max-w-lg leading-relaxed">
              Analyzing <span className="text-black font-bold underline decoration-primary decoration-4 underline-offset-4">{edaData.total_records.toLocaleString()}</span> road accident records across India to identify critical patterns.
            </p>
          </div>
          
          <div className="relative w-full md:w-[350px] h-[250px] animate-fade-in-up">
            <div className="absolute inset-0 border-2 border-black rounded-3xl overflow-hidden shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] bg-white">
               <Image 
                 src="/dashboard.png" 
                 alt="Analysis Illustration" 
                 fill 
                 className="object-cover opacity-80"
               />
            </div>
            {/* Sticker pop-out */}
            <div className="absolute -top-4 -right-4 fp-sticker bg-primary whitespace-nowrap">
               Live Data Feed
            </div>
          </div>
        </div>
      </section>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 stagger-children">
        <StatCard
          icon={Database}
          label="Total Records"
          value={edaData.total_records.toLocaleString()}
          color="text-blue-400"
          gradient="stat-gradient-blue"
        />
        <StatCard
          icon={Layers}
          label="Features"
          value={edaData.total_features}
          color="text-emerald-400"
          gradient="stat-gradient-emerald"
        />
        <StatCard
          icon={AlertTriangle}
          label="Fatal Accidents"
          value={fatalCount.toLocaleString()}
          color="text-red-400"
          gradient="stat-gradient-red"
        />
        <StatCard
          icon={Users}
          label="Serious Injuries"
          value={seriousCount.toLocaleString()}
          color="text-amber-400"
          gradient="stat-gradient-amber"
        />
      </div>

      {/* Row 1: Severity Pie + Peak Hours Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ChartCard
          title="Accident Severity Distribution"
          description="Class imbalance visualization"
          icon={PieChartIcon}
          accentColor="red"
        >
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie
                data={edaData.severity_distribution}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={120}
                innerRadius={70}
                paddingAngle={4}
                stroke="#000"
                strokeWidth={2}
                label={({ name, percent }: { name: string; percent: number }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
              >
                {edaData.severity_distribution.map((entry, index) => {
                   const colors: Record<string, string> = {
                     "Fatal injury": "#e05a47",
                     "Serious Injury": "#d4a843",
                     "Slight Injury": "#6bc4b3"
                   };
                   return <Cell key={index} fill={colors[entry.name] || entry.color} />;
                })}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Peak Accident Hours"
          description="Accidents by hour of day"
          icon={Clock}
          accentColor="amber"
        >
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={edaData.peak_hours} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
              <defs>
                <linearGradient id="hourGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#d4a843" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#d4a843" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" vertical={false} />
              <XAxis dataKey="hour" tick={axisTickStyle} interval={2} angle={-45} textAnchor="end" height={60} axisLine={{stroke: '#000'}} />
              <YAxis tick={axisTickStyle} axisLine={{stroke: '#000'}}>
                <RechartsLabel value="Number of Accidents" angle={-90} position="insideLeft" offset={10} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase', letterSpacing: '1px' }} />
              </YAxis>
              <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: '#000', strokeWidth: 1 }} />
              <Area type="monotone" dataKey="count" stroke="#d4a843" strokeWidth={3} fill="url(#hourGradient)" name="Accidents" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 2: Day Distribution + Weather Heatmap */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ChartCard
          title="Accidents by Day of Week"
          description="Weekly distribution pattern"
          icon={Calendar}
          accentColor="violet"
        >
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={edaData.day_distribution} margin={{ top: 10, right: 30, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" vertical={false} />
              <XAxis 
                dataKey="day" 
                tick={axisTickStyle} 
                interval={0} 
                angle={-45} 
                textAnchor="end" 
                height={50}
                axisLine={{stroke: '#000'}} 
              />
              <YAxis tick={axisTickStyle} axisLine={{stroke: '#000'}}>
                <RechartsLabel value="Frequency" angle={-90} position="insideLeft" offset={10} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase', letterSpacing: '1px' }} />
              </YAxis>
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
              <Bar
                dataKey="count"
                fill="#8c729c"
                radius={[4, 4, 0, 0]}
                stroke="#000"
                strokeWidth={1.5}
                name="Accidents"
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Weather vs Severity Heatmap"
          description="Cross-tabulation of weather conditions and severity"
          icon={CloudRain}
          accentColor="blue"
        >
          <WeatherHeatmapChart data={edaData.weather_severity} />
        </ChartCard>
      </div>

      {/* Row 3: Vehicle Type + Top Causes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ChartCard
          title="Vehicle Types Involved"
          description="Distribution by vehicle type"
          icon={Car}
          accentColor="cyan"
        >
          <ResponsiveContainer width="100%" height={450}>
            <BarChart
              data={edaData.vehicle_type_distribution}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" horizontal={false} />
              <XAxis type="number" tick={axisTickStyle} axisLine={{stroke: '#000'}} height={50}>
                <RechartsLabel value="Accident Count" position="bottom" offset={0} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }} />
              </XAxis>
              <YAxis
                dataKey="name"
                type="category"
                tick={{ ...axisTickStyle, fontSize: 10 }}
                width={130}
                interval={0}
                axisLine={{stroke: '#000'}}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
              <Bar
                dataKey="count"
                fill="#6bc4b3"
                radius={[0, 4, 4, 0]}
                stroke="#000"
                strokeWidth={1.5}
                name="Accidents"
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Top 10 Causes of Accidents"
          description="Leading accident causes"
          icon={Flame}
          accentColor="rose"
        >
          <ResponsiveContainer width="100%" height={450}>
            <BarChart
              data={edaData.cause_distribution}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" horizontal={false} />
              <XAxis type="number" tick={axisTickStyle} axisLine={{stroke: '#000'}} height={50}>
                <RechartsLabel value="Accident Count" position="bottom" offset={0} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }} />
              </XAxis>
              <YAxis
                dataKey="name"
                type="category"
                tick={{ ...axisTickStyle, fontSize: 10 }}
                width={200}
                interval={0}
                axisLine={{stroke: '#000'}}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
              <Bar
                dataKey="count"
                fill="#e05a47"
                radius={[0, 4, 4, 0]}
                stroke="#000"
                strokeWidth={1.5}
                name="Accidents"
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 4: Accident Dynamics - Collision & Pedestrian */}
      <h3 className="text-xl font-serif font-bold text-black flex items-center gap-3 mt-12 mb-6">
        <div className="h-8 w-8 rounded-lg bg-black text-white flex items-center justify-center">
          <Activity className="h-4 w-4" />
        </div>
        Accident Dynamics
      </h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ChartCard
          title="Collision Type Distribution"
          description="Specific nature of vehicle impact"
          icon={Zap}
          accentColor="cyan"
        >
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={edaData.collision_distribution}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" horizontal={false} />
              <XAxis type="number" tick={axisTickStyle} axisLine={{stroke: '#000'}} height={50}>
                <RechartsLabel value="Accident Count" position="bottom" offset={0} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }} />
              </XAxis>
              <YAxis
                dataKey="name"
                type="category"
                tick={{ ...axisTickStyle, fontSize: 10 }}
                width={150}
                interval={0}
                axisLine={{stroke: '#000'}}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
              <Bar
                dataKey="count"
                fill="#4FD1C5"
                radius={[0, 4, 4, 0]}
                stroke="#000"
                strokeWidth={1.5}
                name="Accidents"
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Pedestrian Involvement"
          description="Proportion of accidents involving pedestrians"
          icon={Users}
          accentColor="emerald"
        >
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie
                data={edaData.pedestrian_distribution}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                innerRadius={60}
                paddingAngle={8}
                stroke="#000"
                strokeWidth={2}
                label={({ name, percent }: { name: string; percent: number }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
              >
                <Cell fill="#6bc4b3" />
                <Cell fill="#e2e8f0" />
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 5: Impact Analysis - Regression Target */}
      <h3 className="text-xl font-serif font-bold text-black flex items-center gap-3 mt-12 mb-6">
        <div className="h-8 w-8 rounded-lg bg-black text-white flex items-center justify-center">
          <TrendingUp className="h-4 w-4" />
        </div>
        Impact & Severity Analysis
      </h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ChartCard
          title="Casualty Frequency"
          description="Distribution of Number of Casualties (Regression Target)"
          icon={Activity}
          accentColor="rose"
        >
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={edaData.casualty_distribution} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" vertical={false} />
              <XAxis dataKey="count" tick={axisTickStyle} label={{ value: 'Casualties', position: 'insideBottom', offset: -5, fontSize: 10, fontWeight: 'bold' }} axisLine={{stroke: '#000'}} />
              <YAxis tick={axisTickStyle} axisLine={{stroke: '#000'}} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
              <Bar
                dataKey="occurrences"
                fill="#F56565"
                radius={[4, 4, 0, 0]}
                stroke="#000"
                strokeWidth={1.5}
                name="Incidents"
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Driver Age vs Accident Severity"
          description="Age group breakdown by severity"
          icon={Users}
          accentColor="amber"
        >
          <AgeSeverityChart data={edaData.age_severity} />
        </ChartCard>
      </div>

      {/* Row 6: Situational Context */}
      <h3 className="text-xl font-serif font-bold text-black flex items-center gap-3 mt-12 mb-6">
        <div className="h-8 w-8 rounded-lg bg-black text-white flex items-center justify-center">
          <Layers className="h-4 w-4" />
        </div>
        Situational Context
      </h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ChartCard
          title="Vehicle Defects Distribution"
          description="Reported mechanical issues prior to accident"
          icon={AlertTriangle}
          accentColor="rose"
        >
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={edaData.defect_distribution}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" horizontal={false} />
              <XAxis type="number" tick={axisTickStyle} axisLine={{stroke: '#000'}} />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ ...axisTickStyle, fontSize: 10 }}
                width={150}
                axisLine={{stroke: '#000'}}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
              <Bar
                dataKey="count"
                fill="#ED8936"
                radius={[0, 4, 4, 0]}
                stroke="#000"
                strokeWidth={1.5}
                name="Accidents"
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Accident Area Distribution"
          description="Analysis of surrounding infrastructure"
          icon={MapPin}
          accentColor="blue"
        >
          <ResponsiveContainer width="100%" height={350}>
            <BarChart
              data={edaData.area_distribution}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" horizontal={false} />
              <XAxis type="number" tick={axisTickStyle} axisLine={{stroke: '#000'}} />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ ...axisTickStyle, fontSize: 10 }}
                width={180}
                axisLine={{stroke: '#000'}}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
              <Bar
                dataKey="count"
                fill="#4299E1"
                radius={[0, 4, 4, 0]}
                stroke="#000"
                strokeWidth={1.5}
                name="Accidents"
              />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

// ─── 2. CLASSIFICATION MODELS ──────────────────────────────────────

function ClassificationModels({ data }: { data: ClassModel[] }) {
  const [approachFilter, setApproachFilter] = useState<string>("all");
  const [tableSortMetric, setTableSortMetric] = useState<"f1Score" | "accuracy" | "precision" | "recall" | "rocAuc">("f1Score");
  const [visibleMetrics, setVisibleMetrics] = useState({
    F1: true,
    Accuracy: true,
    Precision: true,
    Recall: true,
  });

  const toggleMetric = (metric: keyof typeof visibleMetrics) => {
    setVisibleMetrics((prev) => ({ ...prev, [metric]: !prev[metric] }));
  };

  const [hiddenRadarModels, setHiddenRadarModels] = useState<Record<string, boolean>>({});
  const toggleRadarModel = (modelName: string) => {
    setHiddenRadarModels((prev) => ({ ...prev, [modelName]: !prev[modelName] }));
  };

  const approaches = ["all", "base", "smote", "smote_xgboost", "tuned_smote"];
  const approachLabels: Record<string, string> = {
    all: "All Models",
    base: "Base Only",
    smote: "SMOTE Only",
    smote_xgboost: "XGBoost + SMOTE",
    tuned_smote: "Optimized Tuned",
  };

  const filtered =
    approachFilter === "all"
      ? data
      : data.filter((m) => m.approach === approachFilter);

  const sortedFiltered = [...filtered].sort((a, b) => (b[tableSortMetric] as number || 0) - (a[tableSortMetric] as number || 0));
  const bestModel = sortedFiltered[0];

  const f1ChartData = filtered.map((m) => ({
    name: m.name.length > 22 ? m.name.slice(0, 20) + "…" : m.name,
    fullName: m.name,
    F1: +(m.f1Score * 100).toFixed(1),
    Accuracy: +(m.accuracy * 100).toFixed(1),
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
  }));

  const topModels = [...sortedFiltered].slice(0, 4);
  const radarData = ["F1", "Accuracy", "Precision", "Recall"].map((metric) => {
    const row: Record<string, string | number> = { metric };
    topModels.forEach((m) => {
      row[m.name.length > 18 ? m.name.slice(0, 16) + "…" : m.name] =
        +((m as any)[metric.toLowerCase() === "f1" ? "f1Score" : metric.toLowerCase()] as number * 100).toFixed(1);
    });
    return row;
  });
  const radarColors = ["#6bc4b3", "#e05a47", "#d4a843", "#4b4b4b"];

  const cm = bestModel?.confusionMatrix;
  const classNames = ["Fatal", "Serious", "Slight"];

  return (
    <div className="space-y-10">
      <div className="flex flex-wrap items-center gap-3 justify-center">
        {approaches.map((a) => (
          <button
            key={a}
            onClick={() => setApproachFilter(a)}
            className={cn(
              "tab-pill font-bold uppercase tracking-wider text-[10px]",
              approachFilter === a ? "tab-pill-active" : "tab-pill-inactive"
            )}
          >
            {approachLabels[a]}
          </button>
        ))}
      </div>

      {!bestModel ? (
        <div className="flex flex-col items-center justify-center p-20 border-2 border-dashed border-black/20 rounded-3xl bg-primary/5">
          <div className="h-16 w-16 bg-black/5 rounded-full flex items-center justify-center mb-4">
            <ShieldAlert className="h-8 w-8 text-black/20" />
          </div>
          <h3 className="text-xl font-serif font-bold text-black/40">No classification models found</h3>
          <p className="text-sm text-black/30 mt-2">Try selecting a different filter above.</p>
          <button 
            onClick={() => setApproachFilter("all")}
            className="mt-6 tab-pill tab-pill-active text-[10px] font-bold uppercase tracking-widest bg-black text-white"
          >
            Reset Filters
          </button>
        </div>
      ) : (
        <>
          <div className="relative">
            <CustomPathway d="M 0 50 Q 100 0 200 50 T 400 50" className="h-32 -top-16 left-0 opacity-10" />
            <Card className="border-black border-2 shadow-[8px_8px_0px_0px_rgba(107,196,179,1)] bg-white relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 opacity-10 pointer-events-none">
                 <Image src="/hero-illustration.png" alt="Decoration" fill className="object-cover" />
              </div>
              <CardContent className="flex items-center gap-6 p-8 relative z-10">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-black text-primary border-2 border-black shadow-[4px_4px_0px_0px_rgba(107,196,179,1)]">
                  <Trophy className="h-8 w-8" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-2xl font-serif font-bold text-foreground">{bestModel.name}</h3>
                    <div className="fp-sticker bg-primary">Best Performance</div>
                  </div>
                  <p className="text-md text-muted-foreground font-medium">
                    Achieving <span className="text-black font-bold">{(bestModel.f1Score * 100).toFixed(1)}% F1-Score</span> using the <span className="italic">{bestModel.approach.toUpperCase()}</span> architectural approach.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <ChartCard
              title="Metric Overview"
              description="Top 4 models radar comparison"
              icon={Target}
              accentColor="violet"
            >
              <div className="flex flex-wrap gap-2 mb-4 justify-center">
                {topModels.map((m, i) => {
                  const isVisible = !hiddenRadarModels[m.name];
                  const shortName = m.name.length > 18 ? m.name.slice(0, 16) + "…" : m.name;
                  return (
                    <button
                      key={m.name}
                      onClick={() => toggleRadarModel(m.name)}
                      className={cn(
                        "px-3 py-1 text-[10px] font-bold uppercase tracking-widest rounded-full border-2 transition-all flex items-center gap-2",
                        isVisible ? "bg-black text-white border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,0.5)]" : "bg-transparent text-black/40 border-black/20 hover:border-black/50"
                      )}
                    >
                      <div className="h-2 w-2 rounded-full" style={{ backgroundColor: isVisible ? radarColors[i] : '#ccc' }} />
                      {shortName} {isVisible ? "✓" : "+"}
                    </button>
                  );
                })}
              </div>
              <div className="h-[450px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={150}>
                    <PolarGrid stroke="#000" opacity={0.1} />
                    <PolarAngleAxis dataKey="metric" tick={{ ...axisTickStyle, fontSize: 12 }} />
                    <PolarRadiusAxis tick={{ fill: "#000", fontSize: 10, opacity: 0.3 }} domain={[0, 100]} axisLine={false} />
                    {topModels.map((m, i) => {
                      if (hiddenRadarModels[m.name]) return null;
                      const modelKey = m.name.length > 18 ? m.name.slice(0, 16) + "…" : m.name;
                      return (
                        <Radar
                          key={m.name}
                          name={modelKey}
                          dataKey={modelKey}
                          stroke={radarColors[i]}
                          fill={radarColors[i]}
                          fillOpacity={0.1}
                          strokeWidth={3}
                        />
                      );
                    })}
                    <Legend 
                      wrapperStyle={{ paddingTop: 20, fontWeight: 700, fontSize: 12, textTransform: 'uppercase' }} 
                      verticalAlign="bottom"
                    />
                    <Tooltip contentStyle={tooltipStyle} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            <ChartCard
              title="Class Performance Breakdown"
              description="Comparison of Accuracy, Precision, Recall & F1 (%)"
              icon={BarChart3}
              accentColor="emerald"
            >
              <div className="flex flex-wrap gap-2 mb-4 justify-center">
                {["F1", "Accuracy", "Precision", "Recall"].map((m) => {
                  const mk = m as keyof typeof visibleMetrics;
                  const colorMap = { F1: "#6bc4b3", Accuracy: "#4a6fa5", Precision: "#d4a843", Recall: "#e05a47" };
                  return (
                    <button
                      key={mk}
                      onClick={() => toggleMetric(mk)}
                      className={cn(
                        "px-3 py-1 text-[10px] font-bold uppercase tracking-widest rounded-full border-2 transition-all flex items-center gap-2",
                        visibleMetrics[mk] ? "bg-black text-white border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,0.5)]" : "bg-transparent text-black/40 border-black/20 hover:border-black/50"
                      )}
                    >
                      <div className="h-2 w-2 rounded-full" style={{ backgroundColor: visibleMetrics[mk] ? colorMap[mk] : '#ccc' }} />
                      {mk} {visibleMetrics[mk] ? "✓" : "+"}
                    </button>
                  );
                })}
              </div>
              <div className="overflow-y-auto overflow-x-hidden pr-2 h-[450px] scrollbar-hide border border-black/5 rounded-xl">
                <div style={{ height: `${Math.max(450, f1ChartData.length * 60)}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={f1ChartData} layout="vertical" margin={{ left: 20, right: 30, top: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" horizontal={false} />
                      <XAxis type="number" tick={axisTickStyle} domain={[0, 100]} axisLine={{stroke: '#000'}} height={50}>
                        <RechartsLabel value="Score (%)" position="bottom" offset={0} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }} />
                      </XAxis>
                      <YAxis
                        dataKey="name"
                        type="category"
                        tick={{ ...axisTickStyle, fontSize: 11 }}
                        width={160}
                        axisLine={{stroke: '#000'}}
                      />
                      <Tooltip contentStyle={tooltipStyle} formatter={(val: number) => `${val}%`} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
                      {visibleMetrics.F1 && <Bar dataKey="F1" fill="#6bc4b3" radius={[0, 4, 4, 0]} stroke="#000" strokeWidth={0.5} />}
                      {visibleMetrics.Accuracy && <Bar dataKey="Accuracy" fill="#4a6fa5" radius={[0, 4, 4, 0]} stroke="#000" strokeWidth={0.5} />}
                      {visibleMetrics.Precision && <Bar dataKey="Precision" fill="#d4a843" radius={[0, 4, 4, 0]} stroke="#000" strokeWidth={0.5} />}
                      {visibleMetrics.Recall && <Bar dataKey="Recall" fill="#e05a47" radius={[0, 4, 4, 0]} stroke="#000" strokeWidth={0.5} />}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </ChartCard>
          </div>

          <div className="fp-card bg-white border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-black text-white shrink-0">
                  <Layers className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-xl font-serif font-bold text-black">Detailed Metric Scorecard</h3>
                  <p className="text-xs font-bold uppercase tracking-widest text-black/40">Performance across all models</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {[ "f1Score", "accuracy", "precision", "recall", "rocAuc" ].map(m => {
                  const mk = m as "f1Score" | "accuracy" | "precision" | "recall" | "rocAuc";
                  return (
                    <button
                      key={mk}
                      onClick={() => setTableSortMetric(mk)}
                      className={cn(
                        "px-3 py-1 text-[10px] font-bold uppercase tracking-widest rounded-full border-2 transition-all",
                        tableSortMetric === mk ? "bg-black text-white border-black" : "bg-transparent text-black/40 border-black/20 hover:border-black/50"
                      )}
                    >
                      {mk === "f1Score" ? "Overall (F1)" : mk === "rocAuc" ? "ROC-AUC" : mk}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-separate border-spacing-y-1">
                <thead>
                  <tr className="border-b-2 border-black">
                    <th className="text-left py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Model Name</th>
                    <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Type</th>
                    <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Accuracy</th>
                    <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Precision</th>
                    <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Recall</th>
                    <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">F1 Score</th>
                    <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedFiltered.map((model) => {
                    const isBest = model.name === bestModel.name;
                    return (
                      <tr
                        key={model.name}
                        className={cn(
                          "group transition-all hover:translate-x-1",
                          isBest ? "bg-primary/10" : "hover:bg-secondary/20"
                        )}
                      >
                        <td className="py-4 px-3 font-bold border-y border-transparent">
                          <div className="flex items-center gap-2">
                            {isBest && <Star className="h-4 w-4 text-black fill-primary" />}
                            <span className={cn(isBest && "text-black")}>{model.name}</span>
                          </div>
                        </td>
                        <td className="py-4 px-3 text-center">
                          <Badge className="bg-white border-2 border-black text-black text-[9px] font-bold">
                            {model.approach}
                          </Badge>
                        </td>
                        <td className="py-4 px-3 text-center font-medium">{(model.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-4 px-3 text-center font-medium">{(model.precision * 100).toFixed(1)}%</td>
                        <td className="py-4 px-3 text-center font-medium">{(model.recall * 100).toFixed(1)}%</td>
                        <td className="py-4 px-3 text-center font-bold text-lg">{(model.f1Score * 100).toFixed(1)}%</td>
                        <td className="py-4 px-3 text-center text-black/50 font-bold">
                          {model.rocAuc ? (model.rocAuc * 100).toFixed(1) + "%" : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {cm && (
            <ChartCard
              title={`Confusion Matrix — ${bestModel.name}`}
              description="Relationship between Predicted vs Actual classes"
              icon={ShieldAlert}
              accentColor="red"
            >
              <div className="max-w-xl mx-auto py-4">
                <table className="w-full text-sm border-separate border-spacing-2">
                  <thead>
                    <tr>
                      <th className="p-3" />
                      {classNames.map((c) => (
                        <th key={c} className="text-center p-2 text-[10px] font-bold uppercase tracking-widest text-black/40">
                          Pred: {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {classNames.map((actual, rowIdx) => (
                      <tr key={actual}>
                        <td className="p-2 text-[10px] font-bold uppercase tracking-widest text-black/40 text-right">Actual: {actual}</td>
                        {cm[rowIdx].map((val, colIdx) => {
                          const maxVal = Math.max(...cm.flat());
                          const intensity = val / maxVal;
                          const isCorrect = rowIdx === colIdx;
                          return (
                            <td key={colIdx} className="p-1">
                              <div
                                className={cn(
                                  "rounded-2xl h-20 flex items-center justify-center text-xl font-serif font-bold border-2 transition-all hover:scale-[1.03]",
                                  isCorrect
                                    ? "bg-primary border-black text-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]"
                                    : val > 0
                                      ? "bg-white border-black/10 text-black/30"
                                      : "bg-transparent border-transparent text-black/10"
                                )}
                              >
                                {val}
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="flex justify-center gap-6 mt-8">
                  <div className="flex items-center gap-2">
                     <div className="h-3 w-3 rounded-full bg-primary border border-black" />
                     <span className="text-[10px] font-bold uppercase tracking-widest text-black/60">Correct Prediction</span>
                  </div>
                  <div className="flex items-center gap-2">
                     <div className="h-3 w-3 rounded-full bg-white border border-black/10" />
                     <span className="text-[10px] font-bold uppercase tracking-widest text-black/60">Incorrect Prediction</span>
                  </div>
                </div>
              </div>
            </ChartCard>
          )}
        </>
      )}
    </div>
  );
}


// ─── 3. REGRESSION MODELS ──────────────────────────────────────────

function RegressionModels({ data }: { data: RegModel[] }) {
  const [approachFilter, setApproachFilter] = useState<string>("all");
  const [tableSortMetric, setTableSortMetric] = useState<"r2" | "mae" | "rmse" | "mse">("r2");

  const approaches = ["all", "base", "pca"];
  const approachLabels: Record<string, string> = {
    all: "All Models",
    base: "Base Only",
    pca: "PCA Reduced",
  };

  const filtered =
    approachFilter === "all"
      ? data
      : data.filter((m) => {
          if (approachFilter === "base") return m.approach === "(Native Features)";
          if (approachFilter === "pca") return m.approach === "(PCA Reduced)";
          return m.approach === approachFilter;
        });

  const sortedFiltered = [...filtered].sort((a, b) => {
    if (tableSortMetric === "r2") return b.r2 - a.r2; // Higher is better
    return (a[tableSortMetric] as number) - (b[tableSortMetric] as number); // Lower is better
  });

  const bestModel = sortedFiltered[0];

  const r2Data = filtered.map((m) => ({
    name: m.name.length > 25 ? m.name.slice(0, 23) + "…" : m.name,
    fullName: m.name,
    R2: +(m.r2 * 100).toFixed(1),
  }));

  const maeData = filtered.map((m) => ({
    name: m.name.length > 25 ? m.name.slice(0, 23) + "…" : m.name,
    fullName: m.name,
    MAE: +m.mae.toFixed(4),
    RMSE: +m.rmse.toFixed(4),
  }));

  if (!bestModel) {
    return (
      <div className="flex flex-col items-center justify-center p-20 border-2 border-dashed border-black/20 rounded-3xl bg-secondary/5">
        <div className="h-16 w-16 bg-black/5 rounded-full flex items-center justify-center mb-4">
          <Activity className="h-8 w-8 text-black/20" />
        </div>
        <h3 className="text-xl font-serif font-bold text-black/40">No models found for this approach</h3>
        <p className="text-sm text-black/30 mt-2">Try selecting a different filter above.</p>
        <button 
          onClick={() => setApproachFilter("all")}
          className="mt-6 tab-pill tab-pill-active text-[10px] font-bold uppercase tracking-widest"
        >
          Reset Filters
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      {/* Filter Buttons - Pill Style */}
      <div className="flex flex-wrap items-center gap-3 justify-center">
        {approaches.map((a) => (
          <button
            key={a}
            onClick={() => setApproachFilter(a)}
            className={cn(
              "tab-pill font-bold uppercase tracking-wider text-[10px]",
              approachFilter === a ? "tab-pill-active" : "tab-pill-inactive"
            )}
          >
            {approachLabels[a] || a}
          </button>
        ))}
      </div>

      {/* Best Model Highlight */}
      <div className="fp-card bg-secondary/30 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-10">
           <TrendingUp className="h-24 w-24 rotate-12" />
        </div>
        <div className="flex flex-col md:flex-row items-center gap-8 relative z-10">
          <div className="flex h-20 w-20 shrink-0 items-center justify-center rounded-3xl bg-black text-secondary border-4 border-black shadow-[4px_4px_0px_0px_rgba(235,222,197,1)]">
            <Trophy className="h-10 w-10" />
          </div>
          <div className="flex-1 text-center md:text-left">
            <div className="flex flex-wrap justify-center md:justify-start items-center gap-2 mb-2">
              <span className="text-xs font-bold uppercase tracking-widest text-black/50">Top Regression Model</span>
              <Badge className="bg-black text-secondary border-none text-[10px] font-bold px-2 py-0">BEST R²</Badge>
            </div>
            <h3 className="text-3xl font-serif font-bold text-black mb-2">{bestModel.name}</h3>
            <p className="text-sm font-medium text-black/60 max-w-xl">
              Provides the most accurate casualty estimations by minimizing prediction error. Evaluated using R² score, Mean Absolute Error (MAE), and RMSE.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4">
             <div className="text-center p-3 rounded-2xl border-2 border-black bg-white shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]">
                <span className="block text-[10px] uppercase font-bold text-black/40">R² Score</span>
                <span className="text-xl font-bold">{(bestModel.r2 * 100).toFixed(1)}%</span>
             </div>
             <div className="text-center p-3 rounded-2xl border-2 border-black bg-white shadow-[3px_3px_0px_0px_rgba(0,0,0,1)]">
                <span className="block text-[10px] uppercase font-bold text-black/40">MAE</span>
                <span className="text-xl font-bold">{bestModel.mae.toFixed(4)}</span>
             </div>
          </div>
        </div>
      </div>

      {/* R² Chart */}
      <ChartCard
        title="R² Score Comparison"
        description="Predictive power across models (%)"
        icon={TrendingUp}
        accentColor="emerald"
      >
        <ResponsiveContainer width="100%" height={600}>
          <BarChart data={r2Data} margin={{ top: 10, right: 30, left: 0, bottom: 120 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" vertical={false} />
            <XAxis 
              dataKey="name" 
              tick={axisTickStyle} 
              interval={0} 
              angle={-45} 
              textAnchor="end" 
              height={100}
              axisLine={{stroke: '#000'}} 
            />
            <YAxis tick={axisTickStyle} domain={[0, "auto"]} axisLine={{stroke: '#000'}}>
              <RechartsLabel value="Score (%)" angle={-90} position="insideLeft" offset={15} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }} />
            </YAxis>
            <Tooltip contentStyle={tooltipStyle} formatter={(val: number) => `${val}%`} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
            <Bar dataKey="R2" fill="#6bc4b3" radius={[4, 4, 0, 0]} stroke="#000" strokeWidth={1.5} name="R² Score" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* MAE / RMSE Chart */}
      <ChartCard
        title="Error Metrics Comparison"
        description="Mean Absolute Error & Root Mean Squared Error (Lower is better)"
        icon={AlertTriangle}
        accentColor="red"
      >
        <ResponsiveContainer width="100%" height={600}>
          <BarChart data={maeData} margin={{ top: 10, right: 30, left: 0, bottom: 120 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ebdec5" vertical={false} />
            <XAxis 
              dataKey="name" 
              tick={axisTickStyle} 
              interval={0} 
              angle={-45} 
              textAnchor="end" 
              height={100}
              axisLine={{stroke: '#000'}} 
            />
            <YAxis tick={axisTickStyle} axisLine={{stroke: '#000'}}>
              <RechartsLabel value="Metric Value" angle={-90} position="insideLeft" offset={15} style={{ textAnchor: 'middle', fill: '#000', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }} />
            </YAxis>
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(0,0,0,0.05)' }} />
            <Legend wrapperStyle={{ paddingTop: 40, fontWeight: 700, fontSize: 11, textTransform: 'uppercase' }} />
            <Bar dataKey="MAE" fill="#d4a843" radius={[4, 4, 0, 0]} stroke="#000" strokeWidth={1.5} name="MAE" />
            <Bar dataKey="RMSE" fill="#e05a47" radius={[4, 4, 0, 0]} stroke="#000" strokeWidth={1.5} name="RMSE" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Metrics Table */}
      <div className="fp-card bg-white border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-black text-white shrink-0">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-xl font-serif font-bold text-black">Regression Scorecard</h3>
              <p className="text-xs font-bold uppercase tracking-widest text-black/40">Model accuracy for casualty prediction</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {[ "r2", "mae", "rmse", "mse" ].map(m => {
              const mk = m as "r2" | "mae" | "rmse" | "mse";
              return (
                <button
                  key={mk}
                  onClick={() => setTableSortMetric(mk)}
                  className={cn(
                    "px-3 py-1 text-[10px] font-bold uppercase tracking-widest rounded-full border-2 transition-all",
                    tableSortMetric === mk ? "bg-black text-white border-black" : "bg-transparent text-black/40 border-black/20 hover:border-black/50"
                  )}
                >
                  {mk === "r2" ? "Overall (R²)" : mk.toUpperCase()}
                </button>
              );
            })}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-separate border-spacing-y-1">
            <thead>
              <tr className="border-b-2 border-black">
                <th className="text-left py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Model Name</th>
                <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Approach</th>
                <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">MAE</th>
                <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">RMSE</th>
                <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">R² Score</th>
              </tr>
            </thead>
            <tbody>
              {sortedFiltered.map((model) => {
                const isBest = model.name === bestModel.name;
                return (
                  <tr
                    key={model.name}
                    className={cn(
                      "group transition-all hover:translate-x-1",
                      isBest ? "bg-secondary/20" : "hover:bg-muted/30"
                    )}
                  >
                    <td className="py-4 px-3 font-bold">
                      <div className="flex items-center gap-2">
                        {isBest && <Star className="h-4 w-4 text-black fill-[#d4a843]" />}
                        <span className={cn(isBest && "text-black")}>{model.name}</span>
                      </div>
                    </td>
                    <td className="py-4 px-3 text-center">
                      <Badge className="bg-white border-2 border-black text-black text-[9px] font-bold">
                        {model.approach}
                      </Badge>
                    </td>
                    <td className="py-4 px-3 text-center">{model.mae.toFixed(4)}</td>
                    <td className="py-4 px-3 text-center">{model.rmse.toFixed(4)}</td>
                    <td className="py-4 px-3 text-center font-bold text-lg">{model.r2.toFixed(4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ─── 4. MODEL PLAYGROUND TAB ───────────────────────────────────────

function ModelPlaygroundTab() {
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTask, setActiveTask] = useState<"classification" | "regression" | "auxiliary">("classification");

  useEffect(() => {
    fetch("/api/model-comparison")
      .then((r) => r.json())
      .then((data) => {
        setModelData(data);
        setLoading(false);
      })
      .catch((e) => {
        console.error(e);
        setLoading(false);
      });
  }, []);

  if (loading) return <ModelSkeleton />;
  if (!modelData)
    return (
      <div className="fp-card border-destructive text-destructive font-bold text-center py-10">
        Failed to load model data. Ensure backend is running.
      </div>
    );

  return (
    <div className="space-y-10">
      {/* Task Toggle - Pill Style */}
      <div className="flex flex-wrap items-center justify-center gap-4">
        <button
          onClick={() => setActiveTask("classification")}
          className={cn(
            "tab-pill flex items-center gap-3 font-bold uppercase tracking-wider text-[11px]",
            activeTask === "classification" ? "tab-pill-active" : "tab-pill-inactive"
          )}
        >
          <Target className="h-4 w-4" />
          Classification (Severity)
          {activeTask === "classification" && <span className="ml-1">→</span>}
        </button>
        <button
          onClick={() => setActiveTask("regression")}
          className={cn(
            "tab-pill flex items-center gap-3 font-bold uppercase tracking-wider text-[11px]",
            activeTask === "regression" ? "tab-pill-active" : "tab-pill-inactive"
          )}
        >
          <TrendingUp className="h-4 w-4" />
          Regression (Casualties)
          {activeTask === "regression" && <span className="ml-1">→</span>}
        </button>
        <button
          onClick={() => setActiveTask("auxiliary")}
          className={cn(
            "tab-pill flex items-center gap-3 font-bold uppercase tracking-wider text-[11px]",
            activeTask === "auxiliary" ? "tab-pill-active" : "tab-pill-inactive"
          )}
        >
          <Activity className="h-4 w-4" />
          Auxiliary Models
          {activeTask === "auxiliary" && <span className="ml-1">→</span>}
        </button>
      </div>

      {activeTask === "classification" && (
        <ClassificationModels data={modelData.classification} />
      )}
      {activeTask === "regression" && (
        <RegressionModels data={modelData.regression} />
      )}
      {activeTask === "auxiliary" && modelData.auxiliary && (
        <AuxiliaryModels data={modelData.auxiliary} />
      )}
    </div>
  );
}

// ─── 4.5 AUXILIARY MODELS ─────────────────────────────────────────

function AuxiliaryModels({ data }: { data: Record<string, Record<string, any>> }) {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="text-center py-20 text-black/40">
        <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p className="font-medium">No auxiliary tasks found. Train the models first.</p>
      </div>
    );
  }

  return (
    <div className="space-y-12 animate-fade-in">
      {Object.entries(data).map(([taskName, models]) => {
        const isClustering = taskName.includes("Clustering") || taskName.includes("Driver Profile");

        return (
          <div key={taskName} className="fp-card bg-white border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
            <h3 className="text-xl font-serif font-bold text-black mb-6 flex items-center gap-3 border-b-2 border-black/5 pb-4">
              <Layers className="h-5 w-5 text-primary" />
              {taskName}
            </h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-separate border-spacing-y-1">
                <thead>
                  <tr className="border-b-2 border-black">
                    <th className="text-left py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Model Name</th>
                    {isClustering ? (
                      <>
                        <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Clusters</th>
                        <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Davies-Bouldin</th>
                        <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Silhouette</th>
                      </>
                    ) : (
                      <>
                        <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Accuracy</th>
                        <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Precision</th>
                        <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">Recall</th>
                        <th className="text-center py-4 px-3 text-black font-bold text-[10px] uppercase tracking-widest">F1-Score</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(models).map(([modelName, metrics]) => (
                    <tr key={modelName} className="hover:bg-muted/30 transition-colors">
                      <td className="py-4 px-3 font-bold text-black">{modelName}</td>
                      {isClustering ? (
                        <>
                          <td className="py-4 px-3 text-center">{metrics.n_clusters}</td>
                          <td className="py-4 px-3 text-center">{metrics.davies_bouldin?.toFixed(4)}</td>
                          <td className="py-4 px-3 text-center font-bold text-lg">{metrics.silhouette_score?.toFixed(4)}</td>
                        </>
                      ) : (
                        <>
                          <td className="py-4 px-3 text-center font-medium">
                            {metrics.accuracy ? (metrics.accuracy * 100).toFixed(1) + "%" : "—"}
                          </td>
                          <td className="py-4 px-3 text-center font-medium">
                            {metrics.precision_weighted ? (metrics.precision_weighted * 100).toFixed(1) + "%" : "—"}
                          </td>
                          <td className="py-4 px-3 text-center font-medium">
                            {metrics.recall_weighted ? (metrics.recall_weighted * 100).toFixed(1) + "%" : "—"}
                          </td>
                          <td className="py-4 px-3 text-center font-bold text-lg">
                            {metrics.f1_weighted ? (metrics.f1_weighted * 100).toFixed(1) + "%" : "—"}
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── 5. SEVERITY RESULT ────────────────────────────────────────────

function SeverityResult({
  data,
}: {
  data: {
    severity: string;
    confidence: number;
    probabilities: Record<string, number> | null;
    model: string;
  };
}) {
  const severityColorMap: Record<string, string> = {
    "Slight Injury": "bg-[#6bc4b3] text-black border-black",
    "Serious Injury": "bg-[#d4a843] text-black border-black",
    "Fatal injury": "bg-[#e05a47] text-white border-black",
  };

  const probBarColor: Record<string, string> = {
    "Slight Injury": "bg-[#6bc4b3]",
    "Serious Injury": "bg-[#d4a843]",
    "Fatal injury": "bg-[#e05a47]",
  };

  const colorClass = severityColorMap[data.severity] || "bg-secondary text-black border-black";
  const confidenceValue = data.confidence != null ? +(data.confidence * 100).toFixed(1) : 0;

  return (
    <div className="space-y-8">
      {/* Predicted Severity Badge + Confidence Gauge */}
      <div className="flex flex-col items-center gap-6 py-6">
        <div className={cn(
          "inline-flex items-center gap-4 rounded-full border-[3px] px-10 py-5 text-2xl font-serif font-bold shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] transition-all",
          colorClass,
        )}>
          <ShieldAlert className="h-8 w-8" />
          {data.severity}
        </div>
        <div className="flex items-center gap-6">
          <ConfidenceGauge
            value={confidenceValue}
            label="Model Confidence"
            color={
              data.severity === "Fatal injury" ? "stroke-[#e05a47]" :
              data.severity === "Serious Injury" ? "stroke-[#d4a843]" :
              "stroke-[#6bc4b3]"
            }
          />
        </div>
      </div>

      {/* Probability Bars */}
      {data.probabilities && (
        <div className="space-y-4 fp-card bg-secondary/10 border-black/10">
          <p className="text-[10px] font-bold uppercase tracking-widest text-black/40">
            Class Distribution
          </p>
          <div className="space-y-4">
            {Object.entries(data.probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([cls, prob]) => (
                <div key={cls} className="space-y-2">
                  <div className="flex justify-between items-end">
                    <span className="text-sm font-bold text-black/70">{cls}</span>
                    <span className="font-serif font-bold text-lg">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-3 bg-white border border-black/10 rounded-full overflow-hidden">
                    <div
                      className={cn("h-full rounded-full transition-all duration-1000 ease-out border-r border-black/20", probBarColor[cls] || "bg-primary")}
                      style={{ width: `${Math.max(prob * 100, 1)}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="flex items-center justify-center gap-3 py-4 border-t-2 border-dashed border-black/5">
        <div className="flex items-center gap-2 px-3 py-1 bg-black text-white rounded-full text-[10px] font-bold uppercase tracking-widest">
          <Info className="h-3 w-3" />
          Model: {data.model}
        </div>
      </div>
    </div>
  );
}

// ─── 6. CASUALTY RESULT ────────────────────────────────────────────

function CasualtyResult({
  data,
}: {
  data: {
    predicted_casualties: number;
    predicted_casualties_float: number;
    model: string;
  };
}) {
  const count = data.predicted_casualties;
  const color =
    count === 0
      ? "text-[#6bc4b3]"
      : count <= 2
        ? "text-[#d4a843]"
        : "text-[#e05a47]";
  
  const bg =
    count === 0
      ? "bg-[#6bc4b3] border-black"
      : count <= 2
        ? "bg-[#d4a843] border-black"
        : "bg-[#e05a47] border-black text-white";

  return (
    <div className="space-y-8">
      <div className="flex flex-col items-center gap-6 py-6">
        <div className={cn(
          "inline-flex items-center gap-5 rounded-3xl border-[3px] px-10 py-6 transition-all shadow-[10px_10px_0px_0px_rgba(0,0,0,1)]",
          bg,
        )}>
          <Users className="h-10 w-10" />
          <span className="text-6xl font-serif font-bold">
            {count}
          </span>
        </div>
        <div className="text-center">
          <p className="text-sm font-bold text-black/60">
            Estimated Casualties
          </p>
          <p className="text-[10px] font-bold uppercase tracking-widest text-black/40 mt-1">
            Exact Regression: <span className="text-black">{data.predicted_casualties_float.toFixed(4)}</span>
          </p>
        </div>
        
        {/* Person icons for visual representation */}
        <div className="flex items-center gap-2 mt-2">
          {Array.from({ length: Math.min(count, 8) }).map((_, i) => (
            <div key={i} className={cn("h-6 w-6 rounded-full border-2 border-black flex items-center justify-center", bg)}>
              <div className="h-2 w-2 rounded-full bg-current opacity-50" />
            </div>
          ))}
          {count > 8 && (
            <span className="text-sm font-bold ml-1">+{count - 8} others</span>
          )}
        </div>
      </div>

      {/* Visual indicator bar */}
      <div className="space-y-3 fp-card bg-secondary/10 border-black/10">
        <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-black/40">
          <span>Safe (0)</span>
          <span>Moderate (1-2)</span>
          <span>Critical (3+)</span>
        </div>
        <div className="h-4 bg-white border-2 border-black rounded-full overflow-hidden flex shadow-[3px_3px_0px_0px_rgba(0,0,0,0.1)]">
          <div className="bg-[#6bc4b3] flex-[1] border-r border-black/20" />
          <div className="bg-[#d4a843] flex-[2] border-r border-black/20" />
          <div className="bg-[#e05a47] flex-[3]" />
        </div>
        <div className="relative h-4 px-1">
          <div
            className="absolute top-0 w-4 h-4 rounded-full bg-black border-2 border-white shadow-md transition-all duration-500 flex items-center justify-center"
            style={{
              left: `${Math.min(Math.max(count / 6 * 100, 2), 95)}%`,
              transform: 'translateX(-50%)'
            }}
          >
             <div className="h-1 w-1 rounded-full bg-white" />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-center gap-3 py-4 border-t-2 border-dashed border-black/5">
        <div className="flex items-center gap-2 px-3 py-1 bg-black text-white rounded-full text-[10px] font-bold uppercase tracking-widest">
          <Info className="h-3 w-3" />
          Model: {data.model}
        </div>
      </div>
    </div>
  );
}

// ─── Feature Section Groups ────────────────────────────────────────

const fieldGroups = [
  {
    title: "Driver Information",
    icon: User,
    color: "text-black",
    borderColor: "border-black",
    bgColor: "bg-secondary",
    fields: ["Day_of_week", "Age_band_of_driver", "Sex_of_driver", "Driving_experience"],
  },
  {
    title: "Vehicle & Trip",
    icon: Truck,
    color: "text-black",
    borderColor: "border-black",
    bgColor: "bg-primary",
    fields: ["Type_of_vehicle", "Vehicle_driver_relation", "Service_year_of_vehicle", "Vehicle_movement"],
  },
  {
    title: "Environment",
    icon: MapPin,
    color: "text-black",
    borderColor: "border-black",
    bgColor: "bg-[#ebdec5]",
    fields: ["Area_accident_occured", "Lanes_or_Medians", "Types_of_Junction", "Road_surface_type", "Light_conditions", "Weather_conditions"],
  },
  {
    title: "Incident context",
    icon: AlertTriangle,
    color: "text-black",
    borderColor: "border-black",
    bgColor: "bg-[#e05a47]/20",
    fields: ["Type_of_collision", "Cause_of_accident", "Hour_of_Day"],
  },
];

// ─── 7. LIVE PREDICTOR TAB ─────────────────────────────────────────

function LivePredictorTab() {
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    classification: {
      severity: string;
      confidence: number;
      probabilities: Record<string, number> | null;
      model: string;
    };
    regression: {
      predicted_casualties: number;
      predicted_casualties_float: number;
      model: string;
    };
    auxiliary?: Record<string, { prediction: string; confidence: number | null; error?: string }>;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});

  const toggleSection = (title: string) => {
    setCollapsedSections((prev) => ({ ...prev, [title]: !prev[title] }));
  };

  const handleChange = useCallback(
    (field: string, value: string) => {
      setFormData((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const [classRes, regRes] = await Promise.all([
        fetch("/api/classify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        }),
        fetch("/api/regress", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        }),
      ]);

      const classData = await classRes.json();
      const regData = await regRes.json();

      if (classData.error || regData.error) {
        setError(classData.details || regData.details || classData.error || regData.error);
      } else {
        setResult({
          classification: {
            severity: classData.severity,
            confidence: classData.confidence,
            probabilities: classData.probabilities,
            model: classData.model,
          },
          regression: {
            predicted_casualties: regData.predicted_casualties,
            predicted_casualties_float: regData.predicted_casualties_float,
            model: regData.model,
          },
          auxiliary: classData.auxiliary,
        });
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    }
    setLoading(false);
  };

  const handleReset = () => {
    setFormData({});
    setResult(null);
    setError(null);
  };

  // Count filled fields
  const filledCount = Object.keys(formData).filter((k) => formData[k]).length;
  const totalFields = ALL_FEATURES.length;

  return (
    <div className="space-y-10">
      {/* Form + Results layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        {/* LEFT: Form */}
        <div className="fp-card bg-white border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] flex flex-col h-full">
          <div className="p-8 border-b-2 border-black/5">
            <div className="flex items-center justify-between mb-6">
               <div className="flex items-center gap-3">
                  <div className="h-10 w-10 flex items-center justify-center rounded-xl bg-black text-primary">
                    <Zap className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-serif font-bold text-black">Incident Parameters</h3>
                    <p className="text-[10px] font-bold uppercase tracking-widest text-black/40">Simulation Data Input</p>
                  </div>
               </div>
               <div className="flex flex-col items-end">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-black/40 mb-1">Completion</span>
                  <Badge className="bg-primary text-black border-2 border-black font-bold">
                    {filledCount}/{totalFields} Data points
                  </Badge>
               </div>
            </div>
            <Progress value={(filledCount / totalFields) * 100} className="h-3 border-2 border-black bg-secondary/30" />
          </div>

          <div className="flex-1 overflow-y-auto p-8 space-y-8 max-h-[600px] scrollbar-hide">
            {fieldGroups.map((group) => {
              const GroupIcon = group.icon;
              const isCollapsed = collapsedSections[group.title];
              return (
                <div key={group.title} className="space-y-4">
                  <button
                    onClick={() => toggleSection(group.title)}
                    className="flex items-center gap-3 w-full text-left group/section"
                  >
                    <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg border-2 border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]", group.bgColor)}>
                      <GroupIcon className="h-4 w-4" />
                    </div>
                    <h4 className="text-md font-serif font-bold flex-1 text-black">
                      {group.title}
                    </h4>
                    {isCollapsed ? (
                      <ChevronDown className="h-5 w-5 text-black/20" />
                    ) : (
                      <ChevronUp className="h-5 w-5 text-black" />
                    )}
                  </button>

                  {!isCollapsed && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pl-11">
                      {group.fields.map((key) => {
                        const fieldDef = ALL_FEATURE_OPTIONS[key];
                        if (!fieldDef) return null;
                        const isNumeric = fieldDef.type === "number";
                        return (
                          <div key={key} className="space-y-2">
                            <Label className="text-[10px] font-bold uppercase tracking-widest text-black/40">
                              {fieldDef.label}
                            </Label>
                            {isNumeric ? (
                              <Input
                                type="number"
                                min={fieldDef.min}
                                max={fieldDef.max}
                                placeholder={fieldDef.placeholder || `...`}
                                value={formData[key] || ""}
                                onChange={(e) => handleChange(key, e.target.value)}
                                className="w-full h-11 border-2 border-black bg-[#FFFaf5] rounded-xl font-bold focus:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all"
                              />
                            ) : (
                              <Select
                                value={formData[key] || ""}
                                onValueChange={(v) => handleChange(key, v)}
                              >
                                <SelectTrigger className="w-full h-11 border-2 border-black bg-[#FFFaf5] rounded-xl font-bold focus:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all">
                                  <SelectValue
                                    placeholder={`Select option`}
                                  />
                                </SelectTrigger>
                                <SelectContent className="border-2 border-black rounded-xl">
                                  {fieldDef.options.map((opt: string) => (
                                    <SelectItem key={opt} value={opt} className="font-bold">
                                      {opt}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="p-8 border-t-2 border-black flex gap-4 bg-secondary/10">
            <button
              onClick={handleSubmit}
              disabled={loading || filledCount === 0}
              className="fp-button-active flex-1 bg-black text-primary font-bold uppercase tracking-widest text-xs h-14"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <BrainCircuit className="h-4 w-4 mr-2" />
              )}
              {loading ? "Calculating..." : "Run Intelligence Analysis"}
            </button>
            <button 
              onClick={handleReset} 
              disabled={loading || filledCount === 0}
              className="fp-button-inactive bg-white text-black border-2 border-black font-bold uppercase tracking-widest text-xs h-14 w-32"
            >
              Reset
            </button>
          </div>
        </div>

        {/* RIGHT: Results */}
        <div className="space-y-8">
          {error && (
            <div className="fp-card border-destructive bg-destructive/10 text-destructive shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
              <div className="flex items-center gap-3 mb-2">
                <AlertTriangle className="h-5 w-5" />
                <h4 className="font-bold uppercase tracking-widest text-xs">Analysis Failed</h4>
              </div>
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}

          {result ? (
            <div className="space-y-10 animate-fade-in">
              <div className="fp-card bg-white border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] overflow-hidden">
                <div className="flex items-center gap-3 mb-8 pb-4 border-b-2 border-dashed border-black/5">
                  <div className="h-8 w-8 flex items-center justify-center rounded-lg bg-[#e05a47] text-white">
                    <ShieldAlert className="h-4 w-4" />
                  </div>
                  <h3 className="text-lg font-serif font-bold">Severity Prediction</h3>
                </div>
                <SeverityResult data={result.classification} />
              </div>

              <div className="fp-card bg-white border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] overflow-hidden">
                <div className="flex items-center gap-3 mb-8 pb-4 border-b-2 border-dashed border-black/5">
                  <div className="h-8 w-8 flex items-center justify-center rounded-lg bg-[#d4a843] text-black">
                    <Users className="h-4 w-4" />
                  </div>
                  <h3 className="text-lg font-serif font-bold">Casualty Estimation</h3>
                </div>
                <CasualtyResult data={result.regression} />
              </div>

              {/* Auxiliary Insights Integration */}
              {result.auxiliary && Object.keys(result.auxiliary).length > 0 && (
                <div className="fp-card bg-white border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] overflow-hidden">
                  <div className="flex items-center gap-3 mb-6 pb-4 border-b-2 border-dashed border-black/5">
                    <div className="h-8 w-8 flex items-center justify-center rounded-lg bg-[#6bc4b3] text-black">
                      <Activity className="h-4 w-4" />
                    </div>
                    <h3 className="text-lg font-serif font-bold">Peripheral Context</h3>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(result.auxiliary).map(([taskName, data]) => {
                      if (data.error) return null;
                      return (
                        <div key={taskName} className="p-4 bg-secondary/10 border-2 border-black/10 rounded-xl relative group hover:border-black/30 transition-colors">
                          <p className="text-[10px] font-bold text-black/50 uppercase tracking-widest mb-2">{taskName}</p>
                          <p className="font-bold text-base leading-tight text-black line-clamp-2">
                            {data.prediction}
                          </p>
                          {data.confidence && (
                            <div className="absolute top-4 right-4 bg-white px-2 py-1 rounded border border-black/10 text-[9px] font-bold text-primary shadow-sm">
                              {(data.confidence * 100).toFixed(1)}% Score
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

            </div>
          ) : !error && (
            <div className="fp-card h-full border-black border-dashed bg-secondary/5 flex flex-col items-center justify-center text-center p-12 min-h-[500px]">
               <div className="h-24 w-24 rounded-full border-2 border-black border-dashed flex items-center justify-center mb-8 bg-white/50">
                  <BrainCircuit className="h-10 w-10 text-black/10" />
               </div>
               <h3 className="text-xl font-serif font-bold text-black/30 mb-4">Awaiting Parameters</h3>
               <p className="text-sm font-medium text-black/20 max-w-xs leading-relaxed">
                  Provide crash site details, driver metrics, and environmental data to generate real-time AI safety insights.
               </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── MAIN PAGE ─────────────────────────────────────────────────────

function HomePageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const tabParam = searchParams.get("tab");

  // ── LANDING PAGE ──────────────────────────────────────────────────
  // Rendered when there is NO ?tab= parameter in the URL.
  // Contains ONLY hero, narrative journey, and a CTA footer.
  // The workspace (tabs, charts, predictor) is completely absent here.
  if (!tabParam || tabParam === "home") {
    return (
      <div className="min-h-screen flex flex-col bg-[#FFFaf5] selection:bg-primary/30">
        {/* Floating coin — landing page only */}
        <FloatingSafeRoadsCoin isVisible={true} />

        {/* Hero section */}
        <LandingHero />

        {/* Narrative journey scroll sections */}
        <NarrativeJourney />

        {/* CTA to enter the workspace */}
        <div className="py-24 flex flex-col items-center gap-6 bg-[#FFFaf5]">
          <p className="text-xs font-bold uppercase tracking-[0.3em] text-black/30">
            Ready to analyse?
          </p>
          <button
            onClick={() => router.push("/?tab=dashboard")}
            className="fp-button fp-button-primary !px-12 !py-5 text-sm font-bold uppercase tracking-widest"
          >
            <BarChart3 className="h-5 w-5" />
            Enter Dashboard
          </button>
        </div>

        {/* Footer illustration */}
        <FooterIllustration />

        <footer className="border-t border-border/50 bg-card/20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <ShieldAlert className="h-4 w-4 text-primary" />
                <span>IR-AIS &mdash; India Road Accident Intelligence System</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Built with Next.js, Recharts &amp; Machine Learning
              </p>
            </div>
          </div>
        </footer>
      </div>
    );
  }

  // ── WORKSPACE PAGE ────────────────────────────────────────────────
  // Rendered when ?tab= is present. Contains ONLY the analytics workspace.
  // The landing page (hero, narrative journey, coin) is completely absent here.
  const currentTab = tabParam;

  const handleTabChange = (tab: string) => {
    router.push(`/?tab=${tab}`);
  };

  const workspaceTabs = [
    { id: "dashboard", label: "Dashboard", icon: BarChart3 },
    { id: "models", label: "Model Playground", icon: BrainCircuit },
    { id: "predictor", label: "Live Predictor", icon: Zap },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#FFFaf5] selection:bg-primary/30">
      <main id="main-content" className="flex-1">
        <div className="mx-auto max-w-7xl w-full px-4 sm:px-6 lg:px-8 py-16 pb-16">

          {/* Intelligence Workspace header — models tab only */}
          {currentTab === "models" && (
            <div className="mb-16 text-center animate-fade-in-up">
              <div className="fp-sticker mb-4 mx-auto">
                <Zap className="h-3 w-3 text-primary" />
                <span>Analytics Playground</span>
              </div>
              <h2 className="editorial-title text-5xl md:text-8xl mb-8">
                Intelligence <span className="text-primary italic">Workspace</span>.
              </h2>
              <div className="flex items-center justify-center gap-6 max-w-xl mx-auto">
                <div className="h-0.5 flex-1 bg-black/10" />
                <p className="text-xs font-bold uppercase tracking-[0.3em] text-black/40 px-4 whitespace-nowrap">
                  Select Analysis Vector
                </p>
                <div className="h-0.5 flex-1 bg-black/10" />
              </div>
            </div>
          )}



          <div className="animate-fade-in-up">
            <Suspense fallback={<DashboardSkeleton />}>
              {currentTab === "dashboard" && <DashboardTab />}
              {currentTab === "models" && <ModelPlaygroundTab />}
              {currentTab === "predictor" && <LivePredictorTab />}
            </Suspense>
          </div>
        </div>
      </main>

      {/* Footer illustration */}
      <FooterIllustration />

      <footer className="border-t border-border/50 bg-card/20 mt-auto">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <ShieldAlert className="h-4 w-4 text-primary" />
              <span>IR-AIS &mdash; India Road Accident Intelligence System</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Built with Next.js, Recharts &amp; Machine Learning
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default function HomePage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      }
    >
      <HomePageContent />
    </Suspense>
  );
}

