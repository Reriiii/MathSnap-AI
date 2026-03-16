import { Button } from "@/app/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import {
  Wand2,
  FolderPlus,
  FileCheck,
  FolderOpen,
  AlertCircle,
  MessageSquare,
  TrendingUp,
  Copy,
  ExternalLink
} from "lucide-react";
import { LatexPreview } from "@/app/components/latex-preview";
import { toast } from "sonner";

interface DashboardPageProps {
  onNavigate: (page: string) => void;
  onNewConvert: () => void;
}

export function DashboardPage({ onNavigate, onNewConvert }: DashboardPageProps) {
  // Mock data for the dashboard
  const metrics = [
    {
      icon: FileCheck,
      label: "Total Conversions",
      value: "142",
      trend: "+12 vs last 7 days",
      color: "text-primary"
    },
    {
      icon: FolderOpen,
      label: "Active Projects",
      value: "8",
      trend: "2 new this week",
      color: "text-accent"
    },
    {
      icon: AlertCircle,
      label: "Items in Review",
      value: "23",
      trend: "15 converted / 8 unverified",
      color: "text-warning"
    },
    {
      icon: MessageSquare,
      label: "Chatbot Assists",
      value: "47",
      trend: "+8 vs last 7 days",
      color: "text-chart-2"
    }
  ];

  const recentActivity = [
    {
      id: 1,
      formula: "\\int_0^\\infty e^{-x^2} \\, dx",
      status: "Converted",
      date: "2 hours ago",
      thumbnail: "integral"
    },
    {
      id: 2,
      formula: "\\sum_{n=1}^{\\infty} \\frac{1}{n^2}",
      status: "Edited",
      date: "5 hours ago",
      thumbnail: "sum"
    },
    {
      id: 3,
      formula: "\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}",
      status: "Exported",
      date: "Yesterday",
      thumbnail: "fraction"
    },
    {
      id: 4,
      formula: "\\lim_{x \\to \\infty} f(x)",
      status: "Converted",
      date: "Yesterday",
      thumbnail: "limit"
    }
  ];

  const projects = [
    {
      id: 1,
      name: "Calculus Homework",
      tags: ["Integration", "Derivatives"],
      lastEdited: "1 hour ago",
      progress: 75,
      items: 12
    },
    {
      id: 2,
      name: "Linear Algebra Notes",
      tags: ["Matrices", "Vectors"],
      lastEdited: "3 hours ago",
      progress: 50,
      items: 8
    },
    {
      id: 3,
      name: "Physics Formulas",
      tags: ["Mechanics", "Thermodynamics"],
      lastEdited: "Yesterday",
      progress: 90,
      items: 20
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Converted":
        return "bg-success/10 text-success border-success/20";
      case "Edited":
        return "bg-accent/10 text-accent border-accent/20";
      case "Exported":
        return "bg-primary/10 text-primary border-primary/20";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <div className="min-h-screen pb-20 md:pb-0">
      <div className="container mx-auto px-4 lg:px-6 py-6 lg:py-8 space-y-6 lg:space-y-8">
        {/* Welcome Section */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl lg:text-3xl font-semibold">Welcome back, John Doe</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Continue converting handwritten math into LaTeX
            </p>
          </div>
          <div className="flex gap-2">
            <Button onClick={onNewConvert} className="bg-primary text-primary-foreground">
              <Wand2 className="h-4 w-4 mr-2" />
              New Convert
            </Button>
            <Button variant="outline" onClick={() => toast.info("Projects coming soon!")}>
              <FolderPlus className="h-4 w-4 mr-2" />
              New Project
            </Button>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
          {metrics.map((metric, index) => (
            <Card key={index}>
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">{metric.label}</p>
                    <p className="text-3xl font-semibold">{metric.value}</p>
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <TrendingUp className="h-3 w-3" />
                      {metric.trend}
                    </div>
                  </div>
                  <div className={`p-3 rounded-lg bg-muted`}>
                    <metric.icon className={`h-5 w-5 ${metric.color}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Recent Activity */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Activity</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => toast.info("History page coming soon!")}>
              View All
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivity.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center gap-4 p-3 rounded-lg hover:bg-muted/50 transition-colors"
                >
                  {/* Thumbnail — KaTeX render */}
                  <div className="flex-shrink-0 w-16 h-16 bg-muted rounded-lg flex items-center justify-center overflow-hidden p-1">
                    <LatexPreview latex={item.formula} displayMode={false} className="text-[8px] scale-[0.55] origin-center" />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="mb-1 max-h-6 overflow-hidden">
                      <LatexPreview latex={item.formula} displayMode={false} className="text-sm" />
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className={getStatusColor(item.status)}>
                        {item.status}
                      </Badge>
                      <span className="text-xs text-muted-foreground">{item.date}</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-1">
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <ExternalLink className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Projects Snapshot */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Projects</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => toast.info("Projects page coming soon!")}>
              View All
            </Button>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {projects.map((project) => (
                <Card key={project.id} className="overflow-hidden">
                  <CardContent className="p-4 space-y-3">
                    <div className="space-y-2">
                      <h4 className="font-medium truncate">{project.name}</h4>
                      <div className="flex flex-wrap gap-1">
                        {project.tags.map((tag, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{project.items} items</span>
                        <span>{project.progress}% verified</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2">
                        <div
                          className="bg-primary rounded-full h-2 transition-all"
                          style={{ width: `${project.progress}%` }}
                        />
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-2">
                      <span className="text-xs text-muted-foreground">
                        {project.lastEdited}
                      </span>
                      <Button size="sm" variant="outline" onClick={() => toast.info("Projects coming soon!")}>
                        Continue
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Empty State Example (commented for reference) */}
        {/* 
        <Card>
          <CardContent className="p-12 text-center space-y-4">
            <div className="flex justify-center">
              <div className="bg-muted p-4 rounded-full">
                <Wand2 className="h-8 w-8 text-muted-foreground" />
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">No activity yet</h3>
              <p className="text-sm text-muted-foreground">
                Upload your first image to get started
              </p>
            </div>
            <Button onClick={onNewConvert}>
              <Upload className="h-4 w-4 mr-2" />
              Upload Image
            </Button>
          </CardContent>
        </Card>
        */}
      </div>
    </div>
  );
}
