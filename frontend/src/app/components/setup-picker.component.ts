import { Component, signal, inject } from '@angular/core';
import { SessionStore } from '../session-store';

// Grimoire Console styling
@Component({
  selector: 'app-setup-picker',
  imports: [],
  template: `
    @if (store.session(); as session) {
      <p class="active">Session active: {{ store.sessionId() }}</p>
    } @else {
      <form class="setup" (submit)="$event.preventDefault()">
        <label>
          World
          <select #worldSel (change)="store.selectWorld(worldSel.value)">
            @for (world of store.catalog()?.worlds ?? []; track world.id) {
              <option [value]="world.id" [selected]="world.id === store.selection()?.world?.id">
                {{ world.name }}
              </option>
            }
          </select>
        </label>

        <label>
          Scene
          <select #sceneSel (change)="store.patchSelection({ sceneId: sceneSel.value })">
            @for (scene of store.selection()?.scenes ?? []; track scene.id) {
              <option [value]="scene.id" [selected]="scene.id === store.selection()?.sceneId">
                {{ scene.title }}
              </option>
            }
          </select>
        </label>

        <label>
          Persona
          <select #personaSel (change)="store.patchSelection({ personaId: personaSel.value })">
            @for (persona of store.selection()?.personas ?? []; track persona.id) {
              <option [value]="persona.id" [selected]="persona.id === store.selection()?.personaId">
                {{ persona.name }}
              </option>
            }
          </select>
        </label>

        <label>
          Player name
          <input
            #nameInput
            type="text"
            [value]="playerName()"
            (input)="playerName.set(nameInput.value)"
          />
        </label>

        <button
          type="button"
          (click)="store.createSessionFromSelection(playerName())"
          [disabled]="store.busy() || !store.selection()?.world"
        >
          Start session
        </button>

        @if (store.loadError(); as error) {
          <p class="warning">{{ error }}</p>
        }
      </form>
    }
  `,
  styles: [`
    .setup { display: flex; flex-direction: column; gap: 0.5rem; max-width: 24rem; }
    label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.8rem; color: var(--muted); }
    .active { font-size: 0.9rem; color: var(--muted); }
    .warning { color: var(--danger); font-size: 0.8rem; }
  `],
})
export class SetupPickerComponent {
  readonly store = inject(SessionStore);
  readonly playerName = signal('');
}
