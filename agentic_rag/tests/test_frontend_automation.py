"""
前端自动化测试脚本 - 使用Playwright测试Streamlit界面
"""
from playwright.sync_api import sync_playwright
import time

def test_streamlit_frontend():
    """测试Streamlit前端界面"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # 可视化模式
        page = browser.new_page()
        
        print("=" * 60)
        print("开始测试 Streamlit 前端界面")
        print("=" * 60)
        
        errors = []
        
        try:
            # 1. 访问前端首页
            print("\n[步骤 1] 访问前端首页 http://localhost:8501")
            page.goto("http://localhost:8501")
            page.wait_for_load_state("networkidle", timeout=30)
            time.sleep(2)  # 等待Streamlit完全渲染
            
            # 截图保存
            page.screenshot(path="test_01_homepage.png", full_page=True)
            print("✅ 首页加载成功，已保存截图: test_01_homepage.png")
            
            # 2. 检查页面标题
            print("\n[步骤 2] 检查页面标题")
            title = page.title()
            print(f"页面标题: {title}")
            if "Agentic RAG" in title or "Streamlit" in title:
                print("✅ 标题正确")
            else:
                errors.append(f"页面标题不正确: {title}")
            
            # 3. 检查主要UI元素
            print("\n[步骤 3] 检查主要UI元素")
            
            # 检查侧边栏配置区域
            sidebar = page.locator('[data-testid="stSidebar"], .stSidebar')
            if sidebar.count() > 0:
                print("✅ 侧边栏存在")
            else:
                errors.append("侧边栏未找到")
            
            # 检查配置输入框
            api_inputs = page.locator('input[type="text"], input[type="password"]')
            input_count = api_inputs.count()
            print(f"找到 {input_count} 个输入框")
            
            # 检查按钮
            buttons = page.locator("button")
            button_count = buttons.count()
            print(f"找到 {button_count} 个按钮")
            
            # 4. 填写API配置
            print("\n[步骤 4] 填写API配置")
            
            # 查找API地址输入框并填写
            inputs = page.locator('input')
            for i in range(inputs.count()):
                input_elem = inputs.nth(i)
                placeholder = input_elem.get_attribute("placeholder") or ""
                if "API" in placeholder and "地址" in placeholder:
                    input_elem.fill("http://localhost:8000")
                    print("✅ 已填写API地址")
                    break
            
            # 查找API密钥输入框并填写
            for i in range(inputs.count()):
                input_elem = inputs.nth(i)
                placeholder = input_elem.get_attribute("placeholder") or ""
                if "API" in placeholder and ("密钥" in placeholder or "Key" in placeholder):
                    input_elem.fill("xiaoyi-cs")
                    print("✅ 已填写API密钥")
                    break
            
            page.screenshot(path="test_02_config_filled.png", full_page=True)
            
            # 5. 点击初始化按钮
            print("\n[步骤 5] 点击初始化连接按钮")
            init_buttons = page.locator("button", has_text="初始化")
            if init_buttons.count() > 0:
                init_buttons.first.click()
                print("✅ 已点击初始化按钮")
                time.sleep(5)  # 等待连接检查
                page.screenshot(path="test_03_after_init.png", full_page=True)
            else:
                errors.append("未找到初始化连接按钮")
            
            # 6. 检查聊天输入框
            print("\n[步骤 6] 检查聊天输入框")
            chat_inputs = page.locator('[data-testid="stChatInput"], .stChatInput input, textarea')
            if chat_inputs.count() > 0:
                print("✅ 聊天输入框存在")
                page.screenshot(path="test_04_chat_ready.png", full_page=True)
            else:
                errors.append("聊天输入框未找到")
            
            # 7. 检查模型选择器
            print("\n[步骤 7] 检查模型选择下拉框")
            selects = page.locator("select")
            if selects.count() > 0:
                print(f"✅ 找到 {selects.count()} 个下拉选择框")
            else:
                # 可能是Streamlit原生的selectbox
                selectboxes = page.locator('[data-testid="stSelectbox"]')
                if selectboxes.count() > 0:
                    print(f"✅ 找到 {selectboxes.count()} 个选择框组件")
                else:
                    print("⚠️ 未找到模型选择器")
            
            # 8. 检查高级功能选项
            print("\n[步骤 8] 检查高级功能选项")
            checkboxes = page.locator('[type="checkbox"], [data-testid="stCheckbox"]')
            checkbox_count = checkboxes.count()
            print(f"找到 {checkbox_count} 个复选框")
            
            # 9. 检查文档上传区域
            print("\n[步骤 9] 检查文档上传区域")
            file_uploaders = page.locator('[data-testid="stFileUploader"], input[type="file"]')
            if file_uploaders.count() > 0:
                print("✅ 文档上传区域存在")
            else:
                print("⚠️ 文档上传区域未找到（可能需要展开）")
            
            # 10. 尝试发送测试消息
            print("\n[步骤 10] 尝试发送测试消息")
            chat_input = page.locator('[data-testid="stChatInput"] input')
            if chat_input.count() > 0:
                chat_input.fill("你好，这是一个测试消息")
                print("✅ 已输入测试消息")
                
                # 尝试提交（按Enter或找提交按钮）
                page.keyboard.press("Enter")
                print("✅ 已提交消息")
                time.sleep(2)
                page.screenshot(path="test_05_after_submit.png", full_page=True)
            else:
                errors.append("无法找到聊天输入框")
            
        except Exception as e:
            errors.append(f"测试过程中出现异常: {str(e)}")
            page.screenshot(path="test_error.png", full_page=True)
            print(f"❌ 发生错误: {e}")
        
        # 输出测试结果
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        if errors:
            print(f"\n❌ 发现 {len(errors)} 个问题:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        else:
            print("\n✅ 所有检查通过！")
        
        # 控制台日志检查
        print("\n检查浏览器控制台日志...")
        console_logs = page.evaluate("""
            () => {
                return window.__CONSOLE_LOGS__ || [];
            }
        """)
        
        if console_logs:
            print(f"找到 {len(console_logs)} 条控制台日志")
        else:
            print("无控制台错误")
        
        browser.close()
        
        return len(errors) == 0

if __name__ == "__main__":
    success = test_streamlit_frontend()
    exit(0 if success else 1)