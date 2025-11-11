import os
import sys
import time
import math
import json
import pygame
import pygame_gui
import numpy as np
import astropy.units as u
from astropy.constants import R_sun, M_sun

# -----------------------------
# Ayarlar / Yollar
# -----------------------------
SIDEBAR_WIDTH = 260
VISUAL_SCALE_BASE = 1e-9
MIN_RADIUS_PIXELS = 2

BG_DARK = (16, 22, 34)
PANEL_BG = (20, 28, 44)
PRIMARY = (19, 91, 236)
TEXT_LIGHT = (255, 255, 255)
TEXT_MUTED = (170, 180, 190)
SLIDER_BG = (50, 50, 60)

G_CONST = 6.67430e-11

HOME = os.path.expanduser("~")
DOCUMENTS = os.path.join(HOME, "Documents") if os.path.isdir(os.path.join(HOME, "Documents")) else HOME
APP_SAVE_DIR = os.path.join(DOCUMENTS, "OrbitalSimulator", "Saves")
THUMB_DIR = os.path.join(APP_SAVE_DIR, "thumbnails")
os.makedirs(THUMB_DIR, exist_ok=True)

SAVE_EXT = ".json"

# -----------------------------
# Cisim sınıfları
# -----------------------------
class Star:
    def __init__(self, mass_solar, radius_solar, color, position=(0.0,0.0), velocity=(0.0,0.0)):
        self.mass = float(mass_solar) * M_sun
        self.radius = float(radius_solar) * R_sun
        if isinstance(color, (list,tuple)):
            self.color = tuple(color)
        else:
            try:
                c = pygame.Color(color)
                self.color = (c.r, c.g, c.b)
            except:
                self.color = PRIMARY
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)
        self.trail_world = []

    def draw(self, surf, camera, zoom):
        sx, sy = world_to_screen(self.position, camera, zoom)
        base_px = max(MIN_RADIUS_PIXELS, int((self.radius.to(u.km).value) * VISUAL_SCALE_BASE / 10.0))
        r_px = max(MIN_RADIUS_PIXELS, int(base_px * zoom))
        if len(self.trail_world) > 1:
            pts = [world_to_screen((wx,wy), camera, zoom) for wx,wy in self.trail_world]
            pygame.draw.lines(surf, self.color, False, pts, max(1, int(1*zoom)))
        pygame.draw.circle(surf, self.color, (sx, sy), r_px)
        self.trail_world.append((float(self.position[0]), float(self.position[1])))
        if len(self.trail_world) > 300:
            self.trail_world.pop(0)

    def move(self, dt):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.acceleration[:] = 0.0

class Planet:
    def __init__(self, mass_solar, radius_solar, color, position=(0.0,0.0), velocity=(0.0,0.0)):
        self.mass = float(mass_solar) * M_sun
        self.radius = float(radius_solar) * R_sun * 100
        if isinstance(color, (list,tuple)):
            self.color = tuple(color)
        else:
            try:
                c = pygame.Color(color)
                self.color = (c.r, c.g, c.b)
            except:
                self.color = (255,0,0)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)
        self.trail_world = []

    def draw(self, surf, camera, zoom):
        sx, sy = world_to_screen(self.position, camera, zoom)
        base_px = max(MIN_RADIUS_PIXELS, int((self.radius.to(u.km).value) * VISUAL_SCALE_BASE / 10.0))
        r_px = max(MIN_RADIUS_PIXELS, int(base_px * zoom))
        if len(self.trail_world) > 1:
            pts = [world_to_screen((wx,wy), camera, zoom) for wx,wy in self.trail_world]
            pygame.draw.lines(surf, self.color, False, pts, max(1, int(1*zoom)))
        pygame.draw.circle(surf, self.color, (sx, sy), r_px)
        self.trail_world.append((float(self.position[0]), float(self.position[1])))
        if len(self.trail_world) > 600:
            self.trail_world.pop(0)

    def move(self, dt):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.acceleration[:] = 0.0

# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def circular_velocity_vector(central_star, body_pos):
    r_vec = body_pos - central_star.position
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.array([0.0,0.0])
    speed = math.sqrt(G_CONST * central_star.mass.value / r)
    perp = np.array([-r_vec[1], r_vec[0]])
    normp = np.linalg.norm(perp)
    if normp == 0:
        return np.array([0.0,0.0])
    return (speed * perp / normp)

def world_to_screen(world_pos, camera, zoom):
    wx, wy = world_pos
    x = int(wx * VISUAL_SCALE_BASE * zoom) + SIDEBAR_WIDTH + int(camera[0])
    y = int(wy * VISUAL_SCALE_BASE * zoom) + int(camera[1])
    return (x, y)

def screen_to_world(screen_pos, camera, zoom):
    sx, sy = screen_pos
    wx = (sx - SIDEBAR_WIDTH - camera[0]) / (VISUAL_SCALE_BASE * zoom)
    wy = (sy - camera[1]) / (VISUAL_SCALE_BASE * zoom)
    return (wx, wy)

def parse_color(text):
    try:
        if isinstance(text, (list,tuple)):
            return tuple(text)
        if ',' in text:
            parts = [int(p.strip()) for p in text.split(',')]
            if len(parts) == 3:
                return tuple(parts)
        c = pygame.Color(text)
        return (c.r, c.g, c.b)
    except Exception:
        return PRIMARY

# -----------------------------
# Save/Load + Thumbnail (robust)
# -----------------------------
def list_saved_simulations():
    files = []
    if not os.path.isdir(APP_SAVE_DIR):
        return files
    for fn in os.listdir(APP_SAVE_DIR):
        if fn.lower().endswith(SAVE_EXT):
            path = os.path.join(APP_SAVE_DIR, fn)
            try:
                mtime = os.path.getmtime(path)
            except:
                mtime = 0
            name = os.path.splitext(fn)[0]
            files.append({"name": name, "path": path, "mtime": mtime})
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files

def save_simulation(objects, filename, fullpath=None):
    if fullpath is None:
        os.makedirs(APP_SAVE_DIR, exist_ok=True)
        fullpath = os.path.join(APP_SAVE_DIR, filename + SAVE_EXT)
    payload = []
    for o in objects:
        try:
            radius_solar = (o.radius / R_sun).value
        except Exception:
            radius_solar = float(o.radius / R_sun)
        payload.append({
            "type": "star" if isinstance(o, Star) else "planet",
            "mass_solar": float(o.mass.value / M_sun.value),
            "radius_solar": float(radius_solar),
            "color": list(o.color),
            "position": [float(o.position[0]), float(o.position[1])],
            "velocity": [float(o.velocity[0]), float(o.velocity[1])]
        })
    try:
        with open(fullpath, "w") as f:
            json.dump({"saved_at": time.time(), "objects": payload}, f, indent=2)
        try:
            create_thumbnail_from_save(fullpath)
        except Exception as e:
            print("Thumbnail oluştururken hata oluştu:", e)
        return True, fullpath
    except Exception as e:
        print("Kaydetme hatası:", e)
        return False, None

def load_simulation_from_path(fullpath):
    if not os.path.exists(fullpath):
        return []
    try:
        with open(fullpath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print("Yükleme hatası:", e)
        return []
    if isinstance(data, dict):
        items = data.get("objects", [])
        if not isinstance(items, list):
            if isinstance(data.get("objects"), dict):
                items = [data.get("objects")]
            else:
                items = []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    objs = []
    for e in items:
        if not isinstance(e, dict):
            continue
        typ = e.get("type", "planet")
        m = e.get("mass_solar", 0.001)
        r = e.get("radius_solar", 0.01)
        color = tuple(e.get("color", PRIMARY))
        pos = tuple(e.get("position", (0.0,0.0)))
        vel = tuple(e.get("velocity", (0.0,0.0)))
        if typ == "star":
            s = Star(m, r, color, position=pos, velocity=vel)
            s.trail_world = []
            objs.append(s)
        else:
            p = Planet(m, r, color, position=pos, velocity=vel)
            p.trail_world = []
            objs.append(p)
    return objs

def create_thumbnail_from_save(savepath, thumb_w=320, thumb_h=240):
    try:
        objs = load_simulation_from_path(savepath)
        surf = pygame.Surface((thumb_w, thumb_h))
        surf.fill((10,10,16))
        font = pygame.font.SysFont("Segoe UI", 14)
        if not isinstance(objs, list):
            objs = list(objs)
        if len(objs) == 0:
            text = os.path.splitext(os.path.basename(savepath))[0]
            txt = font.render(text, True, (220,220,220))
            surf.blit(txt, (10, 10))
        else:
            xs = [o.position[0] for o in objs]
            ys = [o.position[1] for o in objs]
            minx, maxx = float(min(xs)), float(max(xs))
            miny, maxy = float(min(ys)), float(max(ys))
            dx = max(1.0, maxx - minx)
            dy = max(1.0, maxy - miny)
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            margin = 0.9
            scale_x = (thumb_w - 40) / (dx * VISUAL_SCALE_BASE) if dx != 0 else 1.0
            scale_y = (thumb_h - 40) / (dy * VISUAL_SCALE_BASE) if dy != 0 else 1.0
            zoom = min(scale_x, scale_y) * margin
            if zoom <= 0 or not math.isfinite(zoom):
                zoom = 1.0
            camera = [thumb_w//2 - SIDEBAR_WIDTH - int(cx * VISUAL_SCALE_BASE * zoom),
                      thumb_h//2 - int(cy * VISUAL_SCALE_BASE * zoom)]
            for o in objs:
                sx = int(o.position[0] * VISUAL_SCALE_BASE * zoom) + SIDEBAR_WIDTH + int(camera[0])
                sy = int(o.position[1] * VISUAL_SCALE_BASE * zoom) + int(camera[1])
                try:
                    base_px = max(1, int((o.radius.to(u.km).value) * VISUAL_SCALE_BASE / 10.0))
                except Exception:
                    base_px = 2
                r_px = max(1, int(base_px * zoom))
                if 0 <= sx < thumb_w and 0 <= sy < thumb_h:
                    pygame.draw.circle(surf, o.color, (sx, sy), max(1, r_px))
                else:
                    clx = max(2, min(thumb_w-2, sx))
                    cly = max(2, min(thumb_h-2, sy))
                    pygame.draw.circle(surf, o.color, (clx, cly), 2)
        base = os.path.splitext(os.path.basename(savepath))[0]
        thumb_path = os.path.join(THUMB_DIR, base + ".png")
        pygame.image.save(surf, thumb_path)
        return thumb_path
    except Exception as ex:
        print("Thumbnail hatası:", repr(ex))
        return None

# -----------------------------
# UI yardımcıları
# -----------------------------
def draw_sidebar(surf, font, screen_h):
    pygame.draw.rect(surf, PANEL_BG, (0,0,SIDEBAR_WIDTH,screen_h))
    surf.blit(font.render("Cosmos Simulator", True, TEXT_LIGHT), (16,12))
    surf.blit(font.render("Simülasyon Araçları", True, TEXT_MUTED), (16,36))

def draw_button_rect(surf, rect, text, font, bg=(40,40,40)):
    pygame.draw.rect(surf, bg, rect, border_radius=8)
    txt = font.render(text, True, TEXT_LIGHT)
    surf.blit(txt, (rect.x + 12, rect.y + 8))

def draw_slider(surf, rect, knob_x, font, speed_multiplier):
    lbl = font.render("Simülasyon Hızı", True, TEXT_LIGHT)
    val = font.render(f"{speed_multiplier:.2f}x", True, TEXT_MUTED)
    surf.blit(lbl, (rect.x, rect.y - 28))
    surf.blit(val, (rect.x + rect.width - 60, rect.y - 28))
    pygame.draw.rect(surf, SLIDER_BG, rect, border_radius=6)
    fill_w = int((knob_x - rect.x))
    if fill_w > 0:
        pygame.draw.rect(surf, PRIMARY, (rect.x, rect.y, fill_w, rect.height), border_radius=6)
    pygame.draw.circle(surf, BG_DARK, (knob_x, rect.y + rect.height//2), 10)
    pygame.draw.circle(surf, PRIMARY, (knob_x, rect.y + rect.height//2), 10, 2)

# -----------------------------
# Main uygulama
# -----------------------------
def main():
    pygame.init()
    pygame.display.set_caption("Cosmos Simulator")
    screen = pygame.display.set_mode((1200, 750), pygame.RESIZABLE)
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 16)
    small_font = pygame.font.SysFont("Segoe UI", 14)

    manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

    # vars
    objects = []
    btn_add_star = pygame.Rect(16, 80, SIDEBAR_WIDTH-32, 38)
    btn_add_planet = pygame.Rect(16, 128, SIDEBAR_WIDTH-32, 38)
    btn_save = pygame.Rect(16, 176, SIDEBAR_WIDTH-32, 36)
    btn_open_menu = pygame.Rect(16, 220, SIDEBAR_WIDTH-32, 36)
    btn_reset = pygame.Rect(16, 264, SIDEBAR_WIDTH-32, 36)
    btn_follow = pygame.Rect(16, 308, SIDEBAR_WIDTH-32, 36)
    slider_rect = pygame.Rect(16, SCREEN_HEIGHT - 110, SIDEBAR_WIDTH-32, 14)
    knob_x = slider_rect.x + slider_rect.width//2

    dialog_window = None
    submit_btn = None
    mass_input = radius_input = color_input = vx_input = vy_input = None
    dialog_type = None
    waiting_for_place = False
    pending_object_data = None

    dragging_knob = False
    speed_multiplier = 1.0
    dt_base = 60*60

    camera = [0.0, 0.0]
    dragging_camera = False
    last_mouse = None
    zoom = 1.0
    follow_pending = False
    follow_target = None

    saved_list = list_saved_simulations()
    thumbs_cache = {}
    def load_thumbs_cache():
        nonlocal thumbs_cache, saved_list
        thumbs_cache = {}
        for s in saved_list:
            try:
                base = s.get("name") if isinstance(s, dict) else str(s)
                thumb_path = os.path.join(THUMB_DIR, base + ".png")
                if not os.path.exists(thumb_path):
                    try:
                        create_thumbnail_from_save(s["path"]) if isinstance(s, dict) else None
                    except Exception as e:
                        print("Thumb oluşturma hatası:", e)
                try:
                    if os.path.exists(thumb_path):
                        img = pygame.image.load(thumb_path).convert_alpha()
                        thumbs_cache[base] = img
                    else:
                        thumbs_cache[base] = None
                except Exception:
                    thumbs_cache[base] = None
            except Exception as ex:
                print("load_thumbs_cache skipping entry:", repr(ex))
    load_thumbs_cache()

    app_state = "menu"  # kesinlikle menü ile başlasın

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.VIDEORESIZE:
                SCREEN_WIDTH, SCREEN_HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                manager.set_window_resolution((SCREEN_WIDTH, SCREEN_HEIGHT))
                slider_rect.y = SCREEN_HEIGHT - 110

            # önce manager işle
            manager.process_events(event)

            if app_state == "menu":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    create_btn = pygame.Rect(SCREEN_WIDTH-220, 20, 180, 40)
                    if create_btn.collidepoint((mx,my)):
                        objects = []
                        camera = [0.0,0.0]; zoom = 1.0; follow_target=None; follow_pending=False
                        app_state = "sim"
                    else:
                        # compute card rects
                        start_x = 24; start_y = 120; gap = 18; card_w = 320; card_h = 220
                        cols = max(1, (SCREEN_WIDTH - 48)//(card_w + gap))
                        x = start_x; y = start_y
                        for s in saved_list:
                            rect = pygame.Rect(x, y, card_w, card_h)
                            load_btn = pygame.Rect(rect.x + card_w//2 - 110, rect.y + card_h - 56, 100, 40)
                            del_btn = pygame.Rect(rect.x + card_w//2 + 10, rect.y + card_h - 56, 100, 40)
                            try:
                                # Load
                                if load_btn.collidepoint((mx,my)):
                                    objects = load_simulation_from_path(s["path"]) if isinstance(s, dict) else []
                                    camera = [0.0,0.0]; zoom = 1.0; follow_target=None; follow_pending=False
                                    app_state = "sim"
                                    break
                                # Delete
                                if del_btn.collidepoint((mx,my)):
                                    # confirmation dialog; ilişkilendir
                                    base = s.get("name") if isinstance(s, dict) else None
                                    conf = pygame_gui.windows.UIConfirmationDialog(
                                        rect=pygame.Rect((SCREEN_WIDTH//2-180, SCREEN_HEIGHT//2-80),(360,160)),
                                        manager=manager,
                                        window_title="Simülasyonu sil",
                                        action_long_desc=f"'{base}' simülasyonunu kalıcı olarak silmek istediğinize emin misiniz?",
                                    )
                                    conf._target_save = s
                                    
                            except Exception as ex:
                                print("menu card tıklama atlandı:", ex)
                            x += card_w + gap
                            if ( (x - start_x) // (card_w + gap) ) % cols == 0:
                                x = start_x; y += card_h + gap
                if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                    print(s)
                    dlg = event.ui_element
                    if hasattr(dlg, "_target_save"):
                        s = dlg._target_save
                        try:
                            path = s.get("path")
                            base = s.get("name")
                            if path and os.path.exists(path):
                                os.remove(path)
                            thumb_path = os.path.join(THUMB_DIR, base + ".png")
                            if os.path.exists(thumb_path):
                                os.remove(thumb_path)
                            saved_list[:] = list_saved_simulations()
                            load_thumbs_cache()
                            pygame_gui.windows.UIMessageWindow(rect=pygame.Rect((SCREEN_WIDTH//2-160, SCREEN_HEIGHT//2-80),(320,120)),
                                                               manager=manager,
                                                               window_title="Silindi",
                                                               html_message=f"Silindi: <b>{base}</b>")
                        except Exception as ex:
                            print("Silme hatası:", ex)
                            pygame_gui.windows.UIMessageWindow(rect=pygame.Rect((SCREEN_WIDTH//2-160, SCREEN_HEIGHT//2-80),(320,120)),
                                                               manager=manager,
                                                               window_title="Silme Başarısız",
                                                               html_message="Dosya silinemedi.")
                # If cancelled, nothing to do (UI handles closing)

            elif app_state == "sim":
                # zoom
                if event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if mx > SIDEBAR_WIDTH:
                        prev_zoom = zoom
                        if event.y > 0:
                            zoom *= 1.15
                        else:
                            zoom /= 1.15
                        zoom = max(0.1, min(6.0, zoom))
                        world_before = screen_to_world((mx,my), camera, prev_zoom)
                        new_screen = world_to_screen(world_before, camera, zoom)
                        camera[0] += (mx - new_screen[0])
                        camera[1] += (my - new_screen[1])

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    knob_rect = pygame.Rect(knob_x-10, slider_rect.y-6, 20, slider_rect.height+12)
                    if knob_rect.collidepoint(event.pos) and mx <= SIDEBAR_WIDTH:
                        dragging_knob = True
                    if (event.button == 2 or event.button == 3) and mx > SIDEBAR_WIDTH:
                        dragging_camera = True
                        last_mouse = event.pos
                        follow_target = None
                        follow_pending = False
                    if event.button == 1:
                        if mx <= SIDEBAR_WIDTH:
                            if btn_add_star.collidepoint((mx,my)) and dialog_window is None:
                                dialog_type = 'star'
                                dlg_w, dlg_h = 420, 260
                                dlg_x = SCREEN_WIDTH//2 - dlg_w//2; dlg_y = SCREEN_HEIGHT//2 - dlg_h//2
                                dialog_window = pygame_gui.elements.UIWindow(manager=manager, rect=pygame.Rect((dlg_x, dlg_y),(dlg_w, dlg_h)), window_display_title="Yıldız Ekle")
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,20),(120,24)), text="Kütle (M☉):", manager=manager, container=dialog_window)
                                mass_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((150,20),(240,28)), manager=manager, container=dialog_window); mass_input.set_text("1.0")
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,60),(120,24)), text="Yarıçap (R☉):", manager=manager, container=dialog_window)
                                radius_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((150,60),(240,28)), manager=manager, container=dialog_window); radius_input.set_text("1.0")
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,100),(120,24)), text="Renk:", manager=manager, container=dialog_window)
                                color_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((150,100),(160,28)), manager=manager, container=dialog_window); color_input.set_text(f"{PRIMARY[0]},{PRIMARY[1]},{PRIMARY[2]}")
                                color_pick_btn = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((320,100),(70,28)), text="Picker", manager=manager, container=dialog_window)
                                submit_btn = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((150,160),(100,40)), text="İleri", manager=manager, container=dialog_window)
                            elif btn_add_planet.collidepoint((mx,my)) and dialog_window is None:
                                dialog_type = 'planet'
                                dlg_w, dlg_h = 480, 320
                                dlg_x = SCREEN_WIDTH//2 - dlg_w//2; dlg_y = SCREEN_HEIGHT//2 - dlg_h//2
                                dialog_window = pygame_gui.elements.UIWindow(manager=manager, rect=pygame.Rect((dlg_x, dlg_y),(dlg_w, dlg_h)), window_display_title="Gezegen Ekle")
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,20),(120,24)), text="Kütle (M☉):", manager=manager, container=dialog_window)
                                mass_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((150,20),(260,28)), manager=manager, container=dialog_window); mass_input.set_text("0.00315")
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,60),(120,24)), text="Yarıçap (R☉):", manager=manager, container=dialog_window)
                                radius_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((150,60),(260,28)), manager=manager, container=dialog_window); radius_input.set_text("0.00915")
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,100),(120,24)), text="Renk:", manager=manager, container=dialog_window)
                                color_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((150,100),(160,28)), manager=manager, container=dialog_window); color_input.set_text("255,80,80")
                                color_pick_btn = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((320,100),(70,28)), text="Picker", manager=manager, container=dialog_window)
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,140),(60,24)), text="Hız X:", manager=manager, container=dialog_window)
                                vx_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((90,140),(120,28)), manager=manager, container=dialog_window); vx_input.set_text("0")
                                pygame_gui.elements.UILabel(relative_rect=pygame.Rect((220,140),(60,24)), text="Hız Y:", manager=manager, container=dialog_window)
                                vy_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((290,140),(120,28)), manager=manager, container=dialog_window); vy_input.set_text("0")
                                submit_btn = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((180,210),(120,40)), text="İleri", manager=manager, container=dialog_window)
                            elif btn_save.collidepoint((mx,my)):
                                dlg_w, dlg_h = 420, 160
                                dlg_x = SCREEN_WIDTH//2 - dlg_w//2; dlg_y = SCREEN_HEIGHT//2 - dlg_h//2
                                sav_dialog = pygame_gui.elements.UIWindow(manager=manager, rect=pygame.Rect((dlg_x, dlg_y),(dlg_w, dlg_h)), window_display_title="Simülasyonu Kaydet")
                                name_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20,20),(120,24)), text="Kaydetme adı:", manager=manager, container=sav_dialog)
                                name_input = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((140,20),(240,28)), manager=manager, container=sav_dialog)
                                name_input.set_text("Simülasyonum")
                                save_confirm = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((140,60),(100,36)), text="Kaydet", manager=manager, container=sav_dialog)
                                save_confirm._name_input = name_input
                                save_confirm._parent_window = sav_dialog
                            elif btn_open_menu.collidepoint((mx,my)):
                                saved_list[:] = list_saved_simulations()
                                load_thumbs_cache()
                                app_state = "menu"
                            elif btn_reset.collidepoint((mx,my)):
                                objects[:] = []
                                camera[:] = [0.0, 0.0]
                                zoom = 1.0
                                follow_target = None; follow_pending = False
                            elif btn_follow.collidepoint((mx,my)):
                                follow_pending = not follow_pending
                                if follow_pending:
                                    follow_target = None
                        else:
                            if follow_pending:
                                best = None; bestd = max(12, int(12 * zoom))
                                for o in objects:
                                    sx, sy = world_to_screen(o.position, camera, zoom)
                                    d = math.hypot(sx - mx, sy - my)
                                    if d < bestd:
                                        best = o; bestd = d
                                if best is not None:
                                    follow_target = best; follow_pending = False
                            elif waiting_for_place:
                                wx, wy = screen_to_world((mx,my), camera, zoom)
                                if pending_object_data is not None:
                                    typ = pending_object_data.get('type'); mass = pending_object_data.get('mass',1.0)
                                    radius = pending_object_data.get('radius',1.0); color = pending_object_data.get('color', PRIMARY)
                                    vx = pending_object_data.get('vx',0.0); vy = pending_object_data.get('vy',0.0)
                                    if typ == 'star':
                                        objects.append(Star(mass, radius, color, position=(wx, wy), velocity=(vx,vy)))
                                    else:
                                        p = Planet(mass, radius, color, position=(wx, wy), velocity=(vx,vy))
                                        if vx==0 and vy==0 and any(isinstance(o, Star) for o in objects):
                                            central = [o for o in objects if isinstance(o, Star)][0]
                                            p.velocity = circular_velocity_vector(central, p.position)
                                        objects.append(p)
                                waiting_for_place = False
                                pending_object_data = None

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (2,3):
                        dragging_camera = False
                        last_mouse = None
                    dragging_knob = False

                if event.type == pygame.MOUSEMOTION:
                    if dragging_camera and last_mouse is not None:
                        dx = event.pos[0] - last_mouse[0]; dy = event.pos[1] - last_mouse[1]
                        camera[0] += dx; camera[1] += dy
                        last_mouse = event.pos
                    if dragging_knob:
                        mx = event.pos[0]
                        knob_x = max(slider_rect.x, min(slider_rect.x + slider_rect.width, mx))
                        rel = (knob_x - slider_rect.x) / slider_rect.width
                        speed_multiplier = 0.1 + rel * 4.9

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    # color picker
                    if dialog_window is not None and hasattr(event.ui_element, "text") and event.ui_element.text == "Picker":
                        pygame_gui.windows.UIColourPickerDialog(rect=pygame.Rect((SCREEN_WIDTH//2-180, SCREEN_HEIGHT//2-120),(360,240)), manager=manager, window_title="Renk Seç")
                    # dialog Next
                    if dialog_window is not None and event.ui_element == submit_btn:
                        try:
                            mass_val = float(mass_input.get_text())
                        except:
                            mass_val = 1.0 if dialog_type == 'star' else 0.003
                        try:
                            radius_val = float(radius_input.get_text())
                        except:
                            radius_val = 1.0 if dialog_type == 'star' else 0.01
                        color_val = parse_color(color_input.get_text())
                        vx_val = vy_val = 0.0
                        if dialog_type == 'planet':
                            try:
                                vx_val = float(vx_input.get_text()); vy_val = float(vy_input.get_text())
                            except:
                                vx_val = vy_val = 0.0
                        pending_object_data = {'type': dialog_type, 'mass': mass_val, 'radius': radius_val, 'color': color_val, 'vx': vx_val, 'vy': vy_val}
                        waiting_for_place = True
                        dialog_window.kill(); dialog_window = None
                        submit_btn = None; mass_input = radius_input = color_input = vx_input = vy_input = None; dialog_type = None
                    # Save confirm
                    if hasattr(event.ui_element, "_name_input") and hasattr(event.ui_element, "_parent_window"):
                        name_input = event.ui_element._name_input
                        parent_window = event.ui_element._parent_window
                        name = name_input.get_text().strip()
                        if name == "":
                            name = f"sim_{int(time.time())}"
                        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_","-")).rstrip()
                        ok, path = save_simulation(objects, safe_name)
                        if ok:
                            try:
                                parent_window.kill()
                            except:
                                pass
                            saved_list[:] = list_saved_simulations()
                            load_thumbs_cache()
                            pygame_gui.windows.UIMessageWindow(rect=pygame.Rect((SCREEN_WIDTH//2-160, SCREEN_HEIGHT//2-80),(320,120)),
                                                               manager=manager,
                                                               window_title="Kaydedildi",
                                                               html_message=f"Kaydedildi: <b>{safe_name}</b>")
                        else:
                            pygame_gui.windows.UIMessageWindow(rect=pygame.Rect((SCREEN_WIDTH//2-160, SCREEN_HEIGHT//2-80),(320,120)),
                                                               manager=manager,
                                                               window_title="Kaydetme Başarısız",
                                                               html_message="Kaydedilemedi. Klasör izinlerini kontrol edin.")
                if event.type == pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED:
                    col = event.colour
                    if dialog_window is not None and color_input is not None:
                        color_input.set_text(f"{col.r},{col.g},{col.b}")

                # Confirmation dialog events (Delete confirmed)
                if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                    print(s)
                    dlg = event.ui_element
                    if hasattr(dlg, "_target_save"):
                        s = dlg._target_save
                        try:
                            path = s.get("path")
                            base = s.get("name")
                            if path and os.path.exists(path):
                                os.remove(path)
                            thumb_path = os.path.join(THUMB_DIR, base + ".png")
                            if os.path.exists(thumb_path):
                                os.remove(thumb_path)
                            saved_list[:] = list_saved_simulations()
                            load_thumbs_cache()
                            pygame_gui.windows.UIMessageWindow(rect=pygame.Rect((SCREEN_WIDTH//2-160, SCREEN_HEIGHT//2-80),(320,120)),
                                                               manager=manager,
                                                               window_title="Silindi",
                                                               html_message=f"Silindi: <b>{base}</b>")
                        except Exception as ex:
                            print("Delete error:", ex)
                            pygame_gui.windows.UIMessageWindow(rect=pygame.Rect((SCREEN_WIDTH//2-160, SCREEN_HEIGHT//2-80),(320,120)),
                                                               manager=manager,
                                                               window_title="Silme Başarısız",
                                                               html_message="Dosya silinemedi.")
                # If cancelled, nothing to do (UI handles closing)

        # manager update
        manager.update(time_delta)

        # physics
        n = len(objects)
        for o in objects:
            o.acceleration[:] = 0.0
        for i in range(n):
            for j in range(i+1, n):
                a = objects[i]; b = objects[j]
                r_vec = b.position - a.position
                dist = np.linalg.norm(r_vec)
                if dist == 0: continue
                f = G_CONST * a.mass.value * b.mass.value / (dist*dist)
                dir_uv = r_vec / dist
                a.acceleration += (f / a.mass.value) * dir_uv
                b.acceleration -= (f / b.mass.value) * dir_uv
        for o in objects:
            o.move(dt_base * speed_multiplier)

        if follow_target is not None:
            tx, ty = world_to_screen(follow_target.position, camera, zoom)
            center_x = SIDEBAR_WIDTH + (SCREEN_WIDTH - SIDEBAR_WIDTH)//2
            center_y = SCREEN_HEIGHT//2
            camera[0] += (center_x - tx) * 0.12
            camera[1] += (center_y - ty) * 0.12

        # Draw
        if app_state == "menu":
            screen.fill(BG_DARK)
            pygame.draw.rect(screen, (12,12,18), (0,0,SCREEN_WIDTH,80))
            screen.blit(font.render("COSMOS SIMULATOR", True, (240,240,240)), (24,20))
            create_btn_rect = pygame.Rect(SCREEN_WIDTH-220, 20, 180, 40)
            pygame.draw.rect(screen, PRIMARY, create_btn_rect, border_radius=8)
            screen.blit(font.render("Yeni oluştur", True, (255,255,255)), (create_btn_rect.x+10, create_btn_rect.y+8))
            start_x = 24; start_y = 120; gap = 18; card_w = 320; card_h = 220
            cols = max(1, (SCREEN_WIDTH - 48)//(card_w + gap))
            x = start_x; y = start_y
            for s in saved_list:
                rect = pygame.Rect(x, y, card_w, card_h)
                pygame.draw.rect(screen, (18,18,24), rect, border_radius=10)
                base = s.get("name") if isinstance(s, dict) else s
                thumb = thumbs_cache.get(base)
                if thumb:
                    img = pygame.transform.smoothscale(thumb, (card_w, card_h))
                    screen.blit(img, (rect.x, rect.y))
                    overlay = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                    overlay.fill((8,8,12,120))
                    screen.blit(overlay, (rect.x, rect.y))
                screen.blit(font.render(base, True, (255,255,255)), (rect.x + 12, rect.y + 12))
                mtime = time.localtime(s["mtime"]) if isinstance(s, dict) else time.localtime()
                mt_txt = time.strftime("%Y-%m-%d %H:%M", mtime)
                screen.blit(small_font.render("Son kaydedilen: " + mt_txt, True, (200,200,200)), (rect.x + 12, rect.y + 38))
                load_btn = pygame.Rect(rect.x + card_w//2 - 110, rect.y + card_h - 56, 100, 40)
                del_btn = pygame.Rect(rect.x + card_w//2 + 10, rect.y + card_h - 56, 100, 40)
                pygame.draw.rect(screen, PRIMARY, load_btn, border_radius=8)
                pygame.draw.rect(screen, (200,50,50), del_btn, border_radius=8)
                screen.blit(font.render("Yükle", True, (255,255,255)), (load_btn.x + 30, load_btn.y + 8))
                screen.blit(font.render("Sil", True, (255,255,255)), (del_btn.x + 20, del_btn.y + 8))
                x += card_w + gap
                if ( (x - start_x) // (card_w + gap) ) % cols == 0:
                    x = start_x; y += card_h + gap
            manager.draw_ui(screen)
            pygame.display.flip()

        else:
            # sim draw
            screen.fill(BG_DARK)
            draw_sidebar(screen, font, SCREEN_HEIGHT)
            draw_button_rect(screen, btn_add_star, "Yeni Yıldız Ekle", font)
            draw_button_rect(screen, btn_add_planet, "Yeni Gezegen Ekle", font)
            draw_button_rect(screen, btn_save, "Kaydet", font)
            draw_button_rect(screen, btn_open_menu, "Kütüphaneyi Aç", font)
            draw_button_rect(screen, btn_reset, "Sıfırla", font)
            draw_button_rect(screen, btn_follow, "Takip Et (Seç)" if follow_pending else ("Takip Ediliyor" if follow_target else "Takip Et (Seç)"), font)
            slider_rect.y = SCREEN_HEIGHT - 110
            draw_slider(screen, slider_rect, knob_x, small_font, speed_multiplier)
            pygame.draw.rect(screen, BG_DARK, (SIDEBAR_WIDTH, 0, SCREEN_WIDTH - SIDEBAR_WIDTH, SCREEN_HEIGHT))
            for o in objects:
                try:
                    o.draw(screen, camera, zoom)
                except Exception as ex:
                    print("draw object hatası:", ex)
            if waiting_for_place and pending_object_data is not None:
                help_txt = small_font.render("Yerleştirmek için ekrana tıkla", True, TEXT_LIGHT)
                screen.blit(help_txt, (SIDEBAR_WIDTH + 12, SCREEN_HEIGHT - 36))
                mx, my = pygame.mouse.get_pos()
                if mx > SIDEBAR_WIDTH:
                    try:
                        r_px = max(MIN_RADIUS_PIXELS, int(pending_object_data['radius'] * R_sun.to(u.km).value * VISUAL_SCALE_BASE * zoom / 10.0))
                    except Exception:
                        r_px = 4
                    pygame.draw.circle(screen, pending_object_data['color'], (mx, my), r_px, 2)
            manager.draw_ui(screen)
            pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
