// src/components/App.js
import React from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import theme from '../theme';
import Settings from './Settings';
import Chat from './Chat';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div style={{ display: 'flex', height: '100vh', backgroundColor: '#e0e0e0', padding: '20px' }}>
        <div style={{ width: '30%', paddingRight: '10px' }}>
          <Settings />
        </div>
        <div style={{ width: '70%', display: 'flex', justifyContent: 'center', alignItems: 'center', paddingLeft: '10px' }}>
          <Chat />
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;
