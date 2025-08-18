# simple_gun.py   （已去掉声音，已修正玩家‑敌人碰撞）
import pygame
import sys
import random

# ---------- 基础设置 ----------
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("小枪射击（无声版）")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 36)

# ---------- 颜色 ----------
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLACK = (  0,   0,   0)

# ---------- 参数 ----------
PLAYER_SPEED   = 5
BULLET_SPEED   = -10
ENEMY_SPEED    = 2
ENEMY_INTERVAL = 1500          # ms
SHOOT_COOLDOWN = 250           # ms
MAX_LIFE = 3

# ---------- 玩家 ----------
player_pos = pygame.Vector2(WIDTH // 2, HEIGHT - 50)
player_radius = 20
life = MAX_LIFE
score = 0

# ---------- 子弹 ----------
class Bullet(pygame.sprite.Sprite):
    def __init__(self, pos, vel):
        super().__init__()
        self.image = pygame.Surface((6, 12))
        self.image.fill(RED)
        self.rect = self.image.get_rect(center=pos)
        self.velocity = pygame.Vector2(vel)

    def update(self):
        self.rect.move_ip(self.velocity)
        if self.rect.bottom < 0:
            self.kill()

bullets = pygame.sprite.Group()

# ---------- 敌人 ----------
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.velocity = pygame.Vector2(0, ENEMY_SPEED)

    def update(self):
        self.rect.move_ip(self.velocity)
        if self.rect.top > HEIGHT:      # 触底直接让游戏结束
            self.kill()
            global life
            life = 0

enemies = pygame.sprite.Group()

# ---------- 计时器 ----------
ENEMY_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(ENEMY_EVENT, ENEMY_INTERVAL)

last_shot_time = 0
game_over = False

def reset_game():
    global life, score, game_over, last_shot_time
    life = MAX_LIFE
    score = 0
    game_over = False
    last_shot_time = 0
    enemies.empty()
    bullets.empty()
    player_pos.x = WIDTH // 2

# ---------- 主循环 ----------
while True:
    dt = clock.tick(60)          # 维持 60 FPS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if not game_over:
            if event.type == ENEMY_EVENT:
                x = random.randint(0, WIDTH - 30)
                enemies.add(Enemy(x, -30))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                now = pygame.time.get_ticks()
                if now - last_shot_time >= SHOOT_COOLDOWN:
                    bullets.add(Bullet(player_pos, (0, BULLET_SPEED)))
                    last_shot_time = now
        else:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_game()

    if not game_over:
        # ---- 玩家移动 ----
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player_pos.x > player_radius:
            player_pos.x -= PLAYER_SPEED
        if keys[pygame.K_RIGHT] and player_pos.x < WIDTH - player_radius:
            player_pos.x += PLAYER_SPEED

        # ---- 更新精灵 ----
        bullets.update()
        enemies.update()

        # ---- 子弹击中敌人 ----
        hits = pygame.sprite.groupcollide(enemies, bullets, True, True)
        if hits:
            score += len(hits) * 10

        # ---- 敌人碰到玩家（已修正） ----
        player_rect = pygame.Rect(
            int(player_pos.x - player_radius),
            int(player_pos.y - player_radius),
            player_radius * 2,
            player_radius * 2,
        )
        # 检查是否有任意敌人与玩家矩形相交
        if any(e.rect.colliderect(player_rect) for e in enemies):
            life -= 1
            # 删除本帧里所有与玩家相交的敌人，防止一次扣多条命
            for e in list(enemies):
                if e.rect.colliderect(player_rect):
                    e.kill()
            if life <= 0:
                game_over = True

    # ---------- 绘制 ----------
    screen.fill(BLACK)

    # 玩家（圆形）
    pygame.draw.circle(screen, WHITE,
                       (int(player_pos.x), int(player_pos.y)),
                       player_radius)

    # 子弹、敌人
    bullets.draw(screen)
    enemies.draw(screen)

    # UI：分数、生命、FPS
    screen.blit(FONT.render(f"Score: {score}", True, WHITE), (10, 10))
    screen.blit(FONT.render(f"Life: {life}", True, WHITE), (10, HEIGHT - 40))
    screen.blit(FONT.render(f"FPS: {int(clock.get_fps())}", True, WHITE),
                (WIDTH - 120, 10))

    # 游戏结束提示
    if game_over:
        over_surf = FONT.render("GAME OVER – Press R to Restart", True, RED)
        over_rect = over_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(over_surf, over_rect)

    pygame.display.flip()
