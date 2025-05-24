import React, { useState } from 'react';
import './index.css';

function App() {
  const [image, setImage] = useState(null);
  const [style, setStyle] = useState('casual');
  const [response, setResponse] = useState('');

  const handleImageChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setImage(event.target.files[0]);
    }
  };

  const handleStyleChange = (event) => {
    setStyle(event.target.value);
  };

  const handleSubmit = () => {
    // Replace this with actual chatbot API call
    setResponse(`LLM response for ${image ? image.name : 'no image'} with style ${style}`);
  };

  return (
    <div className="app-container">
      <h1 className="title">AI Stylist</h1>

      <div className="upload-section">

        <label className="file-upload">
          {image ? image.name : 'Choose an image'}
          <input type="file" accept="image/*" onChange={handleImageChange} />
        </label>

        <select className="dropdown" value={style} onChange={handleStyleChange}>
          <option value="casual">Casual</option>
          <option value="formal">Formal</option>
          <option value="sporty">Sporty</option>
          <option value="vintage">Vintage</option>
        </select>

        <button className="submit-btn" onClick={handleSubmit}>
          Analyze Style
        </button>
      </div>

      {response && <div className="response-box">{response}</div>}
    </div>
  );
}

export default App;
