import { useState } from "react";
import './Button.css'
import { Link } from "react-router-dom";

function Button(props){
    const [hovering,setIsHovering] = useState({isHover:false,color:"#0ADB5"})
    function handleMouseOver(){
        setIsHovering({isHover:true,color:"#EEEEEE"});
    }
    function handleMouseOut(){
        setIsHovering({isHover:false,color:"#00ADB5"});
    }
    var customStyle={
        backgroundColor:hovering.color
    }
    return(
        <Link to={props.to}>
            <div className="squareBtn" style={customStyle}>
                <img 
                src={props.img} 
                alt={props.alt}
                height={props.height}
                width={props.width} 
                onMouseOver={handleMouseOver}
                onMouseOut={handleMouseOut}
                >   
                </img>
            </div>
            <h2>{props.text}</h2>
        </Link>
    )
}

export default Button
