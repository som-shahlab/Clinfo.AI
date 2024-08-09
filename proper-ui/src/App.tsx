import "./App.css";
import { ThemeProvider } from "./components/theme-provider";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Button } from "./components/ui/button";
import { ModeToggle } from "./components/mode-toggle";

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <div>
        <ModeToggle />
        <Button>Hello world</Button>
      </div>
    ),
  },
]);

function App() {
  return (
    <>
      <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
        <RouterProvider router={router} />
      </ThemeProvider>
    </>
  );
}

export default App;
