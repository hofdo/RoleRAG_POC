import { Component, inject } from '@angular/core';
import { SessionStore } from '../session-store';

// Grimoire Console styling
@Component({
  selector: 'app-transcript',
  imports: [],
  template: `
    @if (store.sessionId() === null) {
      <p class="hint">Start a session to play.</p>
    } @else {
      @if (store.transcript().length === 0) {
        <p class="hint">No turns yet — say something.</p>
      } @else {
        <div class="log">
          @for (entry of store.transcript(); track $index) {
            <div class="row" [class.player]="entry.role === 'player'" [class.assistant]="entry.role === 'assistant'">
              <div class="role">{{ entry.role === 'player' ? 'You' : 'Assistant' }}</div>
              @if (entry.label) {
                <div class="caption">{{ entry.label }}</div>
              }
              <div class="text">{{ entry.text }}</div>
            </div>
          }
          @if (store.busy()) {
            <div class="row assistant thinking is-live streaming">
              <div class="text">…thinking</div>
            </div>
          }
        </div>
      }
    }
  `,
  styles: [`
    .hint { color: var(--muted); font-style: italic; font-family: var(--font-body); }
    .log { display: flex; flex-direction: column; gap: 0.5rem; }
    .row { padding: 0.4rem 0.6rem; border-left: 3px solid var(--border); font-family: var(--font-body); color: var(--text); }
    .row.player { border-left-color: var(--accent); }
    .row.assistant { border-left-color: var(--border); background: var(--surface-2); }
    .role { font-weight: 600; font-size: 0.8rem; color: var(--muted); }
    .caption { font-size: 0.7rem; color: var(--muted); }
    .text { white-space: pre-wrap; }
    .row.assistant:not(.thinking):first-of-type .text::first-letter {
      float: left;
      font-family: var(--font-display);
      color: var(--accent);
      font-size: 3.2em;
      line-height: .78;
      padding-right: 0.06em;
      text-shadow: 0 0 12px color-mix(in srgb, var(--accent) 55%, transparent);
    }
    .thinking .text { color: var(--muted); font-style: italic; }
  `],
})
export class TranscriptComponent {
  readonly store = inject(SessionStore);
}
