import { ModeToggle } from "@/components/mode-toggle";
import { Button } from "@/components/ui/button";
import { useMemo } from "react";

const NAV_ITEMS = [
  {
    label: "Home",
    href: "/",
  },
  {
    label: "Blogs",
    href: "/blogs",
  },
  {
    label: "Pricing",
    href: "/pricing",
  },
];

const Navbar = () => {
  const navItems = useMemo(() => {
    // get authstate from localhost
    const isAuth = localStorage.getItem("isAuth");
    if (isAuth) {
      return NAV_ITEMS.concat({
        label: "Dashboard",
        href: "/dashboard",
      });
    } else {
      return NAV_ITEMS.concat({
        label: "Login",
        href: "/auth",
      });
    }
  }, []);
  return (
    <div className="w-fit fixed top-3 left-1/2 -translate-x-1/2">
      <div className="flex justify-center gap-2 bg-primary py-2 px-8 rounded-lg">
        {navItems.map((item, index) => (
          <Button
            key={index + item.href}
            variant={`${
              index === navItems.length - 1 ? "secondary" : "default"
            }`}
            className={`hover:scale-110 transition-all ${
              index === navItems.length - 1 ? "rounded-lg" : ""
            }`}
          >
            {item.label}
          </Button>
        ))}
        <div className="w-[0.1rem] h-auto rounded-ful bg-neutral-700"></div>
        <div className="hover:bg-primary/90 text-primary-foreground">
          <ModeToggle />
        </div>
      </div>
    </div>
  );
};

export default Navbar;
