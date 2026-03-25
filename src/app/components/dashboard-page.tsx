import { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Progress } from "@/app/components/ui/progress";
import { Separator } from "@/app/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/app/components/ui/select";
import { Button } from "@/app/components/ui/button";
import {
  Database, BookOpen, Cpu, Target, Timer,
  LayoutDashboard, TrendingUp, ChevronDown, ChevronRight,
  ArrowRight, Layers, Brain, Zap, Eye, Settings2,
  BarChart3, Activity, GitBranch, Box,
} from "lucide-react";
import { motion } from "motion/react";
import { LatexPreview } from "@/app/components/latex-preview";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from "@/app/components/ui/chart";
import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  LineChart, Line, Area, AreaChart,
  ReferenceLine, ReferenceDot,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

// @ts-ignore
import datasetStats from "@/app/data/dataset-stats.json";
// @ts-ignore
import trainingMetrics from "@/app/data/training-metrics.json";

// ============================================================
// Types
// ============================================================
interface DashboardPageProps {
  onNavigate: (page: string) => void;
  onNewConvert: () => void;
}

// ============================================================
// Constants
// ============================================================
const CHART_COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)",
];

const CATEGORY_COLORS: Record<string, string> = {
  Digits: "#3b82f6",
  Lowercase: "#8b5cf6",
  Uppercase: "#a855f7",
  Greek: "#14b8a6",
  Operators: "#ef4444",
  Functions: "#f59e0b",
  Structural: "#1e3a8a",
  Delimiters: "#10b981",
  Symbols: "#ec4899",
  Other: "#6b7280",
};

const splitConfig: ChartConfig = {
  train: { label: "Training", color: "var(--chart-1)" },
  val: { label: "Validation", color: "var(--chart-2)" },
  test: { label: "Test", color: "var(--chart-3)" },
};

// ============================================================
// Animated Counter Hook
// ============================================================
function useAnimatedCounter(target: number, duration = 1200) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setValue(Math.floor(eased * target));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, duration]);
  return value;
}

// ============================================================
// KPI Card
// ============================================================
function KPICard({
  icon: Icon,
  label,
  value,
  subtitle,
  color,
  delay,
}: {
  icon: any;
  label: string;
  value: string;
  subtitle: string;
  color: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
    >
      <Card className="relative overflow-hidden">
        <div className={`absolute left-0 top-0 bottom-0 w-1 ${color}`} />
        <CardContent className="p-5">
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                {label}
              </p>
              <p className="text-2xl font-bold font-mono tabular-nums">{value}</p>
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            </div>
            <div className="p-2.5 rounded-lg bg-muted">
              <Icon className="h-4 w-4 text-muted-foreground" />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ============================================================
// Overview Tab
// ============================================================
function OverviewTab({ onTabChange }: { onTabChange: (tab: string) => void }) {
  const totalSamples = useAnimatedCounter(datasetStats.totalSamples);
  const vocabSize = useAnimatedCounter(datasetStats.tokenFrequency.length + 4);
  const params = useAnimatedCounter(639, 1000);
  const bestExpRateVal = Math.round(trainingMetrics.bestExpRate * 100);
  const expRate = useAnimatedCounter(bestExpRateVal, 1400);
  const trainedEpochs = trainingMetrics.trainedEpochs;
  const totalEpochs = trainingMetrics.totalEpochs;
  const epochs = useAnimatedCounter(trainedEpochs, 1000);

  const splitDonutData = datasetStats.splits.map((s: any) => ({
    name: s.name,
    value: s.count,
    fill: s.name === "train" ? CHART_COLORS[0] : s.name === "val" ? CHART_COLORS[1] : CHART_COLORS[2],
  }));

  const top5Tokens = datasetStats.tokenFrequency.slice(0, 5).map((t: any) => ({
    token: t.token === "{" ? "\\{" : t.token === "}" ? "\\}" : t.token,
    count: t.total,
    fill: CATEGORY_COLORS[t.category] || CATEGORY_COLORS.Other,
  }));

  const sparklineData = trainingMetrics.epochs
    .filter((_: any, i: number) => i % 5 === 0)
    .map((e: any) => ({ epoch: e.epoch, expRate: e.expRate }));

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        <KPICard icon={Database} label="Total Samples" value={totalSamples.toLocaleString()} subtitle="CROHME 2013/2016/2019" color="bg-blue-500" delay={0} />
        <KPICard icon={BookOpen} label="Vocabulary" value={`${vocabSize}`} subtitle={`${datasetStats.tokenFrequency.length} LaTeX + 4 special`} color="bg-teal-500" delay={0.08} />
        <KPICard icon={Cpu} label="Parameters" value={`${(params / 100).toFixed(2)}M`} subtitle="DenseNet + Transformer" color="bg-purple-500" delay={0.16} />
        <KPICard icon={Target} label="Best ExpRate" value={`${(expRate / 100).toFixed(2)}%`} subtitle={`Epoch ${trainingMetrics.bestEpoch} / ${totalEpochs}`} color="bg-emerald-500" delay={0.24} />
        <KPICard icon={Timer} label="Training" value={`${epochs}/${totalEpochs}`} subtitle={`${((epochs / totalEpochs) * 100).toFixed(1)}% complete`} color="bg-amber-500" delay={0.32} />
      </div>

      {/* Mini Preview Charts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Split Distribution Mini */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
          <Card className="cursor-pointer hover:border-primary/30 transition-colors" onClick={() => onTabChange("dataset")}>
            <CardHeader className="pb-2 pt-4 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs font-medium text-muted-foreground">Dataset Split</CardTitle>
                <ArrowRight className="h-3 w-3 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <ChartContainer config={splitConfig} className="h-[120px] w-full">
                <PieChart>
                  <Pie data={splitDonutData} dataKey="value" nameKey="name" innerRadius={30} outerRadius={50} paddingAngle={2}>
                    {splitDonutData.map((entry: any, i: number) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Pie>
                  <ChartTooltip content={<ChartTooltipContent />} />
                </PieChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </motion.div>

        {/* Top 5 Tokens Mini */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
          <Card className="cursor-pointer hover:border-primary/30 transition-colors" onClick={() => onTabChange("dataset")}>
            <CardHeader className="pb-2 pt-4 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs font-medium text-muted-foreground">Top Tokens</CardTitle>
                <ArrowRight className="h-3 w-3 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <ChartContainer config={{ count: { label: "Frequency", color: "var(--chart-1)" } }} className="h-[120px] w-full">
                <BarChart data={top5Tokens} layout="vertical" margin={{ left: 5, right: 5 }}>
                  <XAxis type="number" hide />
                  <YAxis type="category" dataKey="token" width={30} tick={{ fontSize: 11 }} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {top5Tokens.map((entry: any, i: number) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                  <ChartTooltip content={<ChartTooltipContent />} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </motion.div>

        {/* ExpRate Sparkline */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
          <Card className="cursor-pointer hover:border-primary/30 transition-colors" onClick={() => onTabChange("training")}>
            <CardHeader className="pb-2 pt-4 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs font-medium text-muted-foreground">ExpRate Trend</CardTitle>
                <ArrowRight className="h-3 w-3 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <ChartContainer config={{ expRate: { label: "ExpRate %", color: "var(--chart-2)" } }} className="h-[120px] w-full">
                <AreaChart data={sparklineData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                  <defs>
                    <linearGradient id="expRateGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--chart-2)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="var(--chart-2)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="expRate" stroke="var(--chart-2)" fill="url(#expRateGrad)" strokeWidth={2} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                </AreaChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}

// ============================================================
// Dataset Tab
// ============================================================
function DatasetTab() {
  const [selectedSplit, setSelectedSplit] = useState("all");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [topN, setTopN] = useState(25);

  const filteredTokens = useMemo(() => {
    let tokens = datasetStats.tokenFrequency;
    if (selectedCategory !== "all") {
      tokens = tokens.filter((t: any) => t.category === selectedCategory);
    }
    return tokens.slice(0, topN).map((t: any) => ({
      token: t.token === "{" ? "\\{" : t.token === "}" ? "\\}" : t.token,
      count: selectedSplit === "all" ? t.total : t[selectedSplit],
      category: t.category,
      fill: CATEGORY_COLORS[t.category] || CATEGORY_COLORS.Other,
    }));
  }, [selectedSplit, selectedCategory, topN]);

  const categories = datasetStats.tokenCategories || [];

  const splitDonutData = datasetStats.splits.map((s: any) => ({
    name: s.name,
    value: s.count,
    percentage: s.percentage,
    fill: s.name === "train" ? CHART_COLORS[0] : s.name === "val" ? CHART_COLORS[1] : CHART_COLORS[2],
  }));

  const seqHistogram = datasetStats.sequenceLength?.histogram || [];
  const seqStats = datasetStats.sequenceLength?.stats || [];
  const overallStats = seqStats.find((s: any) => s.split === "all") || {};

  const sourceData = useMemo(() => {
    const sources = datasetStats.datasetSources || [];
    const grouped: Record<string, any> = {};
    sources.forEach((s: any) => {
      if (!grouped[s.source]) grouped[s.source] = { source: s.source, train: 0, val: 0, test: 0 };
      grouped[s.source][s.split] = s.count;
    });
    return Object.values(grouped);
  }, []);

  const complexity = datasetStats.complexityMetrics || {};

  return (
    <div className="space-y-4">
      {/* Filter Bar */}
      <Card>
        <CardContent className="p-3 flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground">Split:</span>
            <Select value={selectedSplit} onValueChange={setSelectedSplit}>
              <SelectTrigger className="h-8 w-[100px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="train">Train</SelectItem>
                <SelectItem value="val">Val</SelectItem>
                <SelectItem value="test">Test</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Separator orientation="vertical" className="h-6" />

          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground">Category:</span>
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="h-8 w-[120px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                {categories.map((c: any) => (
                  <SelectItem key={c.category} value={c.category}>{c.category}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Separator orientation="vertical" className="h-6" />

          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-muted-foreground">Top-N:</span>
            <div className="flex gap-1">
              {[15, 25, 50].map(n => (
                <Button
                  key={n}
                  variant={topN === n ? "default" : "outline"}
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={() => setTopN(n)}
                >
                  {n}
                </Button>
              ))}
            </div>
          </div>

          <div className="ml-auto">
            <Badge variant="secondary" className="text-xs">
              {datasetStats.totalSamples.toLocaleString()} samples
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Row 1: Split + Sources */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Split Distribution */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Split Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={splitConfig} className="h-[220px] w-full">
              <PieChart>
                <Pie data={splitDonutData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={85} paddingAngle={3} label={({ name, percentage }) => `${name} (${percentage}%)`}>
                  {splitDonutData.map((entry: any, i: number) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Pie>
                <ChartTooltip content={<ChartTooltipContent />} />
              </PieChart>
            </ChartContainer>
            <div className="flex justify-center gap-4 mt-2">
              {datasetStats.splits.map((s: any) => (
                <div key={s.name} className="text-center">
                  <p className="text-lg font-bold font-mono">{s.count.toLocaleString()}</p>
                  <p className="text-xs text-muted-foreground capitalize">{s.name}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Dataset Sources */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Dataset Sources (CROHME)</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={splitConfig} className="h-[220px] w-full">
              <BarChart data={sourceData} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="source" tick={{ fontSize: 11 }} width={90} />
                <Bar dataKey="train" stackId="a" fill="var(--chart-1)" radius={[0, 0, 0, 0]} />
                <Bar dataKey="val" stackId="a" fill="var(--chart-2)" />
                <Bar dataKey="test" stackId="a" fill="var(--chart-3)" radius={[0, 4, 4, 0]} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      {/* Row 2: Token Frequency (full width) */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">
              Token Frequency Distribution
              {selectedCategory !== "all" && (
                <Badge variant="outline" className="ml-2 text-xs">{selectedCategory}</Badge>
              )}
            </CardTitle>
            <span className="text-xs text-muted-foreground">
              Showing top {filteredTokens.length} tokens
            </span>
          </div>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{ count: { label: "Frequency", color: "var(--chart-1)" } }}
            className="h-[300px] w-full"
          >
            <BarChart data={filteredTokens} margin={{ bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="token" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" interval={0} height={60} />
              <YAxis tick={{ fontSize: 10 }} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {filteredTokens.map((entry: any, i: number) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
              <ChartTooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-background border rounded-lg px-3 py-2 shadow-lg text-xs space-y-1">
                      <p className="font-mono font-bold">{d.token}</p>
                      <p>Frequency: <span className="font-mono">{d.count.toLocaleString()}</span></p>
                      <p>Category: <span className="font-medium" style={{ color: d.fill }}>{d.category}</span></p>
                    </div>
                  );
                }}
              />
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Row 3: Sequence Length + Categories */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Sequence Length */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sequence Length Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={splitConfig} className="h-[250px] w-full">
              <BarChart data={seqHistogram} margin={{ bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="bin" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" />
                <YAxis tick={{ fontSize: 10 }} />
                <Bar dataKey="train" fill="var(--chart-1)" radius={[2, 2, 0, 0]} />
                <Bar dataKey="val" fill="var(--chart-2)" radius={[2, 2, 0, 0]} />
                <Bar dataKey="test" fill="var(--chart-3)" radius={[2, 2, 0, 0]} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
              </BarChart>
            </ChartContainer>
            {/* Stats summary */}
            <div className="grid grid-cols-4 gap-2 mt-3 text-center">
              {[
                { label: "Min", value: overallStats.min },
                { label: "Mean", value: overallStats.mean },
                { label: "Median", value: overallStats.median },
                { label: "Max", value: overallStats.max },
              ].map(s => (
                <div key={s.label} className="bg-muted/50 rounded-lg p-2">
                  <p className="text-lg font-bold font-mono">{s.value}</p>
                  <p className="text-[10px] text-muted-foreground">{s.label}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Token Categories */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Token Categories</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2.5">
              {categories.map((cat: any) => {
                const maxFreq = Math.max(...categories.map((c: any) => c.totalFrequency));
                const pct = (cat.totalFrequency / maxFreq) * 100;
                return (
                  <button
                    key={cat.category}
                    className="w-full text-left group"
                    onClick={() => setSelectedCategory(cat.category === selectedCategory ? "all" : cat.category)}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-2.5 h-2.5 rounded-sm"
                          style={{ backgroundColor: CATEGORY_COLORS[cat.category] }}
                        />
                        <span className={`text-xs font-medium ${selectedCategory === cat.category ? "text-foreground" : "text-muted-foreground group-hover:text-foreground"} transition-colors`}>
                          {cat.category}
                        </span>
                        <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                          {cat.tokenCount}
                        </Badge>
                      </div>
                      <span className="text-xs font-mono text-muted-foreground">
                        {cat.totalFrequency.toLocaleString()}
                      </span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-1.5">
                      <div
                        className="h-1.5 rounded-full transition-all"
                        style={{
                          width: `${pct}%`,
                          backgroundColor: CATEGORY_COLORS[cat.category],
                          opacity: selectedCategory === "all" || selectedCategory === cat.category ? 1 : 0.3,
                        }}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Row 4: Complexity */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Formula Complexity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            <div className="bg-muted/50 rounded-lg p-3 text-center col-span-1">
              <p className="text-xl font-bold font-mono">{complexity.avgUniqueTokens}</p>
              <p className="text-[10px] text-muted-foreground">Avg Unique Tokens</p>
            </div>
            <div className="bg-muted/50 rounded-lg p-3 text-center col-span-1">
              <p className="text-xl font-bold font-mono">{complexity.avgNestingDepth}</p>
              <p className="text-[10px] text-muted-foreground">Avg Nesting Depth</p>
            </div>
            {(complexity.constructUsage || []).map((c: any) => (
              <div key={c.construct} className="bg-muted/50 rounded-lg p-3 text-center">
                <LatexPreview latex={c.construct} displayMode={false} className="text-sm mb-1" />
                <Progress value={c.percentage} className="h-1.5 mb-1" />
                <p className="text-[10px] text-muted-foreground">
                  {c.percentage}% ({c.count.toLocaleString()})
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ============================================================
// Model Tab
// ============================================================
function ModelTab() {
  const [expandedBlock, setExpandedBlock] = useState<string | null>(null);

  const toggleBlock = (id: string) => {
    setExpandedBlock(expandedBlock === id ? null : id);
  };

  const architectureBlocks = [
    {
      id: "input",
      title: "Input Image",
      icon: Eye,
      color: "bg-blue-500/10 border-blue-500/30 text-blue-600 dark:text-blue-400",
      details: "Grayscale (1 channel) • H: 16-128px • W: 16-512px • Otsu binarization • Scale augmentation (0.7x-1.4x)",
    },
    {
      id: "encoder",
      title: "DenseNet Encoder",
      icon: Layers,
      color: "bg-purple-500/10 border-purple-500/30 text-purple-600 dark:text-purple-400",
      details: "3 Dense Blocks × 16 Bottleneck layers • Growth rate: 24 • Initial Conv: 1→48ch, 7×7, stride 2 • Transition layers (reduction 0.5) • Output: 256-dim feature map • ~4.2M parameters",
    },
    {
      id: "posenc",
      title: "2D Positional Encoding",
      icon: GitBranch,
      color: "bg-teal-500/10 border-teal-500/30 text-teal-600 dark:text-teal-400",
      details: "Sinusoidal 2D encoding • Separate height & width signals • Added to encoder features • d_model: 256",
    },
    {
      id: "decoder",
      title: "Transformer Decoder + ARM",
      icon: Brain,
      color: "bg-amber-500/10 border-amber-500/30 text-amber-600 dark:text-amber-400",
      details: "3 Decoder layers • 8 attention heads • d_model: 256, d_ff: 1024 • ARM: cross_coverage + self_coverage (dc=32) • Dropout: 0.3 • Word embedding + positional encoding • ~1.9M parameters",
    },
    {
      id: "output",
      title: "LaTeX Output",
      icon: Zap,
      color: "bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400",
      details: "Linear projection: 256→114 vocab • Greedy decode (inference) • Beam search: size 10, max_len 200 • ~29K parameters",
    },
  ];

  const paramDistribution = [
    { name: "DenseNet Encoder", value: 4200000, fill: CHART_COLORS[0] },
    { name: "Transformer Decoder", value: 1900000, fill: CHART_COLORS[1] },
    { name: "Positional Encoding", value: 200000, fill: CHART_COLORS[2] },
    { name: "ARM Module", value: 50000, fill: CHART_COLORS[3] },
    { name: "Embeddings + Projection", value: 59000, fill: CHART_COLORS[4] },
  ];

  const paramConfig: ChartConfig = {
    "DenseNet Encoder": { label: "Encoder", color: CHART_COLORS[0] },
    "Transformer Decoder": { label: "Decoder", color: CHART_COLORS[1] },
    "Positional Encoding": { label: "PosEnc", color: CHART_COLORS[2] },
    "ARM Module": { label: "ARM", color: CHART_COLORS[3] },
    "Embeddings + Projection": { label: "Embed/Proj", color: CHART_COLORS[4] },
  };

  const hyperparams = {
    "Data": [
      ["Batch Size", "64", "Samples per training step"],
      ["Max Seq Length", "200", "Maximum LaTeX token output"],
      ["Image Channels", "1", "Grayscale input"],
      ["Height Range", "16 - 128 px", "Input image height bounds"],
      ["Width Range", "16 - 512 px", "Input image width bounds"],
      ["Max Batch Pixels", "4,000,000", "Adaptive batching constraint"],
      ["Augmentation", "Scale 0.7x-1.4x", "Random scaling during training"],
    ],
    "Model": [
      ["d_model", "256", "Model embedding dimension"],
      ["Growth Rate", "24", "DenseNet channels per layer"],
      ["Dense Blocks", "3 × 16 layers", "Encoder depth"],
      ["Attention Heads", "8", "Multi-head attention"],
      ["Decoder Layers", "3", "Transformer decoder depth"],
      ["FFN Dimension", "1024", "Feed-forward hidden size"],
      ["Dropout", "0.3", "Regularization rate"],
      ["ARM dc", "32", "Attention refinement channels"],
      ["Coverage", "Cross + Self", "ARM attention coverage"],
    ],
    "Training": [
      ["Epochs", "300 (trained 230)", "Training duration"],
      ["Optimizer", "Adam", "Gradient descent variant"],
      ["Learning Rate", "1e-4", "Initial learning rate"],
      ["Weight Decay", "1e-4", "L2 regularization"],
      ["LR Scheduler", "ReduceLROnPlateau", "Adaptive LR decay"],
      ["LR Patience", "10 epochs", "Epochs before LR decay"],
      ["LR Factor", "0.5", "LR multiplication factor"],
      ["Grad Clip", "5.0", "Gradient clipping norm"],
      ["Mixed Precision", "Enabled (AMP)", "FP16 training acceleration"],
    ],
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* Architecture Diagram — 3 cols */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Architecture: Mini-CoMER Pipeline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {architectureBlocks.map((block, index) => (
                <motion.div
                  key={block.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.08 }}
                >
                  <button
                    className={`w-full text-left border rounded-lg p-3 transition-all hover:shadow-sm ${block.color}`}
                    onClick={() => toggleBlock(block.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <block.icon className="h-4 w-4" />
                        <span className="text-sm font-medium">{block.title}</span>
                      </div>
                      {expandedBlock === block.id ? (
                        <ChevronDown className="h-3.5 w-3.5" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5" />
                      )}
                    </div>
                    {expandedBlock === block.id && (
                      <motion.p
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        className="text-xs mt-2 opacity-80 leading-relaxed"
                      >
                        {block.details}
                      </motion.p>
                    )}
                  </button>
                  {index < architectureBlocks.length - 1 && (
                    <div className="flex justify-center py-0.5">
                      <div className="w-px h-3 bg-border" />
                    </div>
                  )}
                </motion.div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Parameter Distribution — 2 cols */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Parameter Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={paramConfig} className="h-[200px] w-full">
                <PieChart>
                  <Pie
                    data={paramDistribution}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={45}
                    outerRadius={80}
                    paddingAngle={2}
                  >
                    {paramDistribution.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Pie>
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="bg-background border rounded-lg px-3 py-2 shadow-lg text-xs">
                          <p className="font-medium">{d.name}</p>
                          <p className="font-mono">{(d.value / 1e6).toFixed(2)}M params</p>
                        </div>
                      );
                    }}
                  />
                </PieChart>
              </ChartContainer>
              <div className="text-center mt-2">
                <p className="text-2xl font-bold font-mono">6.39M</p>
                <p className="text-xs text-muted-foreground">Total Parameters</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Hyperparameters */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Settings2 className="h-4 w-4" /> Hyperparameters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(hyperparams).map(([section, rows]) => (
              <div key={section}>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">{section}</h4>
                <div className="space-y-1">
                  {rows.map(([param, value, desc]) => (
                    <div key={param} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50 group transition-colors">
                      <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">{param}</span>
                      <span className="text-xs font-mono font-medium">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ============================================================
// Training Tab
// ============================================================
function TrainingTab() {
  const [showLoss, setShowLoss] = useState(true);
  const [showExpRate, setShowExpRate] = useState(true);
  const [showLR, setShowLR] = useState(true);

  const epochs = trainingMetrics.epochs || [];
  const lrEvents = trainingMetrics.lrEvents || [];
  const bestEpoch = trainingMetrics.bestEpoch || 194;
  const bestExpRate = trainingMetrics.bestExpRate || 47.12;

  // Downsample for performance
  const lossData = epochs.filter((_: any, i: number) => i % 2 === 0);

  const lossConfig: ChartConfig = {
    trainLoss: { label: "Train Loss", color: "var(--chart-1)" },
    valLoss: { label: "Val Loss", color: "var(--chart-5)" },
  };

  const expRateConfig: ChartConfig = {
    expRate: { label: "ExpRate %", color: "var(--chart-2)" },
  };

  const lrConfig: ChartConfig = {
    lr: { label: "Learning Rate", color: "var(--chart-4)" },
  };

  // Summary stats
  const lastEpoch = epochs[epochs.length - 1];
  const bestLossEpoch = epochs.reduce((best: any, e: any) =>
    e.valLoss < best.valLoss ? e : best, epochs[0]);

  return (
    <div className="space-y-4">
      {/* Toggle controls */}
      <Card>
        <CardContent className="p-3 flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">Show:</span>
          {[
            { label: "Loss Curves", active: showLoss, toggle: () => setShowLoss(!showLoss) },
            { label: "ExpRate", active: showExpRate, toggle: () => setShowExpRate(!showExpRate) },
            { label: "LR Schedule", active: showLR, toggle: () => setShowLR(!showLR) },
          ].map(t => (
            <Button
              key={t.label}
              variant={t.active ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={t.toggle}
            >
              {t.label}
            </Button>
          ))}

          <div className="ml-auto flex gap-3">
            <Badge variant="outline" className="text-xs">
              {trainingMetrics.trainedEpochs}/{trainingMetrics.totalEpochs} epochs
            </Badge>
            <Badge className="text-xs bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
              Best: {bestExpRate}%
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Loss Curves */}
      {showLoss && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Loss Curves</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={lossConfig} className="h-[280px] w-full">
              <LineChart data={lossData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" tick={{ fontSize: 10 }} label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} label={{ value: "Loss", angle: -90, position: "insideLeft", fontSize: 10 }} />
                <Line type="monotone" dataKey="trainLoss" stroke="var(--chart-1)" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="valLoss" stroke="var(--chart-5)" strokeWidth={1.5} dot={false} />
                {lrEvents.slice(1).map((ev: any) => (
                  <ReferenceLine key={ev.epoch} x={ev.epoch} stroke="var(--chart-4)" strokeDasharray="4 4" strokeOpacity={0.5} />
                ))}
                <ReferenceLine x={bestEpoch} stroke="var(--chart-3)" strokeDasharray="4 4" label={{ value: "Best", fontSize: 10, fill: "var(--chart-3)" }} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
              </LineChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}

      {/* ExpRate Curve */}
      {showExpRate && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Expression Recognition Rate (ExpRate)</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={expRateConfig} className="h-[280px] w-full">
              <AreaChart data={lossData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                <defs>
                  <linearGradient id="expRateArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--chart-2)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--chart-2)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" tick={{ fontSize: 10 }} label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={[0, 55]} label={{ value: "ExpRate %", angle: -90, position: "insideLeft", fontSize: 10 }} />
                <Area type="monotone" dataKey="expRate" stroke="var(--chart-2)" fill="url(#expRateArea)" strokeWidth={2} dot={false} />
                <ReferenceLine y={bestExpRate} stroke="var(--chart-2)" strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: `${bestExpRate}%`, fontSize: 10, fill: "var(--chart-2)" }} />
                <ReferenceDot x={bestEpoch} y={bestExpRate} r={5} fill="var(--chart-2)" stroke="var(--background)" strokeWidth={2} />
                <ChartTooltip content={<ChartTooltipContent />} />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}

      {/* LR Schedule */}
      {showLR && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Learning Rate Schedule</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={lrConfig} className="h-[200px] w-full">
              <LineChart data={lossData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                <YAxis
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v) => v >= 1e-4 ? "1e-4" : v >= 5e-5 ? "5e-5" : v >= 2.5e-5 ? "2.5e-5" : v >= 1.25e-5 ? "1.25e-5" : "6.25e-6"}
                />
                <Line type="stepAfter" dataKey="lr" stroke="var(--chart-4)" strokeWidth={2} dot={false} />
                <ChartTooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div className="bg-background border rounded-lg px-3 py-2 shadow-lg text-xs">
                        <p>Epoch: <span className="font-mono">{d.epoch}</span></p>
                        <p>LR: <span className="font-mono">{d.lr.toExponential(2)}</span></p>
                      </div>
                    );
                  }}
                />
              </LineChart>
            </ChartContainer>
            {/* LR Events */}
            <div className="flex flex-wrap gap-2 mt-3">
              {lrEvents.map((ev: any, i: number) => (
                <Badge key={i} variant="outline" className="text-[10px]">
                  Ep {ev.epoch}: {ev.newLr.toExponential(1)} — {ev.reason}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold font-mono">{trainingMetrics.trainedEpochs}</p>
            <Progress value={(trainingMetrics.trainedEpochs / trainingMetrics.totalEpochs) * 100} className="h-1.5 mt-2 mb-1" />
            <p className="text-[10px] text-muted-foreground">of {trainingMetrics.totalEpochs} epochs</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold font-mono">{bestLossEpoch?.valLoss?.toFixed(3)}</p>
            <p className="text-xs text-muted-foreground mt-1">Best Val Loss</p>
            <p className="text-[10px] text-muted-foreground">Epoch {bestLossEpoch?.epoch}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold font-mono text-emerald-500">{bestExpRate}%</p>
            <p className="text-xs text-muted-foreground mt-1">Best ExpRate</p>
            <p className="text-[10px] text-muted-foreground">Epoch {bestEpoch}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold font-mono">{lastEpoch?.lr?.toExponential(2)}</p>
            <p className="text-xs text-muted-foreground mt-1">Final LR</p>
            <p className="text-[10px] text-muted-foreground">5 decay events</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ============================================================
// Main Dashboard Page
// ============================================================
export function DashboardPage({ onNavigate, onNewConvert }: DashboardPageProps) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="min-h-screen pb-20 md:pb-0">
      <div className="container mx-auto px-4 lg:px-6 py-4 lg:py-6 space-y-4">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="text-xl lg:text-2xl font-bold flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Mini-CoMER Analytics Dashboard
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              CROHME Dataset · DenseNet + Transformer · Attention Refinement Module
            </p>
          </div>
          <Badge variant="outline" className="text-xs w-fit">
            <Activity className="h-3 w-3 mr-1" />
            {datasetStats.totalSamples.toLocaleString()} samples · {datasetStats.tokenFrequency.length} tokens · 6.39M params
          </Badge>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="overview" className="text-xs gap-1.5">
              <LayoutDashboard className="h-3.5 w-3.5" /> Overview
            </TabsTrigger>
            <TabsTrigger value="dataset" className="text-xs gap-1.5">
              <Database className="h-3.5 w-3.5" /> Dataset
            </TabsTrigger>
            <TabsTrigger value="model" className="text-xs gap-1.5">
              <Box className="h-3.5 w-3.5" /> Model
            </TabsTrigger>
            <TabsTrigger value="training" className="text-xs gap-1.5">
              <TrendingUp className="h-3.5 w-3.5" /> Training
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <OverviewTab onTabChange={setActiveTab} />
          </TabsContent>
          <TabsContent value="dataset">
            <DatasetTab />
          </TabsContent>
          <TabsContent value="model">
            <ModelTab />
          </TabsContent>
          <TabsContent value="training">
            <TrainingTab />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
