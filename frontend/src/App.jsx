import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState } from 'react'
import './App.css'
import Button from './components/Button.jsx'
import Register from './components/Register.jsx'
import Recognise from './components/Recognise.jsx'

function Home() {
  return (
  <div className='btnContainer'>
    <Button img="/add.png" alt="Register" to='/register' text='Register'/>
    <Button img="/register.png" alt="Recognise" to='/recognise' text='Recognise'/>
  </div>
  )
}

function App(){
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/register" element={<Register />} />
        <Route path="/recognise" element={<Recognise />} />
      </Routes>
    </Router>
  );
}


export default App
