import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  // Sigils per view (the design system's mono glyphs): prose, retrieval, analytics, eval.
  readonly nav = [
    { path: '/play', label: 'Play', sigil: '▌' },
    { path: '/inspector', label: 'Inspector', sigil: '⟩' },
    { path: '/analytics', label: 'Analytics', sigil: '∴' },
    { path: '/eval', label: 'Eval', sigil: '⊘' },
  ];
}
