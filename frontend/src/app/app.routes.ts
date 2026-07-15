import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'play', pathMatch: 'full' },
  {
    path: 'play',
    title: 'Play',
    loadComponent: () => import('./pages/play.component').then((m) => m.PlayComponent),
  },
  {
    path: 'inspector',
    title: 'RAG Inspector',
    loadComponent: () => import('./pages/inspector.component').then((m) => m.InspectorComponent),
  },
  {
    path: 'map',
    title: 'Vector Map',
    loadComponent: () => import('./pages/vector-map.component').then((m) => m.VectorMapComponent),
  },
  {
    path: 'analytics',
    title: 'Analytics',
    loadComponent: () => import('./pages/analytics.component').then((m) => m.AnalyticsComponent),
  },
  {
    path: 'eval',
    title: 'Eval',
    loadComponent: () => import('./pages/eval.component').then((m) => m.EvalComponent),
  },
  { path: '**', redirectTo: 'play' },
];
