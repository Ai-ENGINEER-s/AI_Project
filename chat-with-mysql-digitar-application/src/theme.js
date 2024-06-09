// src/theme.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    background: {
      default: '#f0f2f5',
      paper: '#ffffff',
    },
    text: {
      primary: '#000000',
    },
  },
});

export default theme;
