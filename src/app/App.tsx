import { useState } from "react";
import { Header } from "@/app/components/header";
import { HomePage } from "@/app/components/home-page";
import { DashboardPage } from "@/app/components/dashboard-page";
import { ConvertPage } from "@/app/components/convert-page";
import { SettingsPage } from "@/app/components/settings-page";
import { Toaster } from "@/app/components/ui/sonner";

export default function App() {
  const [currentPage, setCurrentPage] = useState("home");

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
    // Scroll to top on navigation
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleNewConvert = () => {
    setCurrentPage("convert");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const renderPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage onNavigate={handleNavigate} onNewConvert={handleNewConvert} />;
      case "dashboard":
        return <DashboardPage onNavigate={handleNavigate} onNewConvert={handleNewConvert} />;
      case "convert":
        return <ConvertPage />;
      case "projects":
        return <ProjectsPage onNavigate={handleNavigate} />;
      case "history":
        return <HistoryPage onNavigate={handleNavigate} />;
      case "templates":
        return <TemplatesPage onNavigate={handleNavigate} />;
      case "settings":
        return <SettingsPage />;
      case "help":
        return <HelpPage />;
      default:
        return <HomePage onNavigate={handleNavigate} onNewConvert={handleNewConvert} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header 
        currentPage={currentPage} 
        onNavigate={handleNavigate}
        onNewConvert={handleNewConvert}
      />
      <main className="w-full">
        {renderPage()}
      </main>
      <Toaster />
    </div>
  );
}
