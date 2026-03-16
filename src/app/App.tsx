import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
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
      case "settings":
        return <SettingsPage />;
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
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPage}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            {renderPage()}
          </motion.div>
        </AnimatePresence>
      </main>
      <Toaster richColors position="top-right" />
    </div>
  );
}
