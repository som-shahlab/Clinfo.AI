import React from "react";
import { Outlet } from "react-router-dom";
import Navbar from "../Navbar";

const Root = () => {
  return (
    <div className="relative w-full h-full">
      <Navbar />
      <main>{<Outlet />}</main>
    </div>
  );
};

export default Root;
