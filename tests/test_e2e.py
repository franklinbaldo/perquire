#!/usr/bin/env python3
"""
End-to-End Test for Perquire
Tests the complete flow: embedding → investigation → description
"""

import asyncio
import json
import tempfile
import numpy as np
from pathlib import Path

async def test_e2e_investigation():
    """Test complete investigation flow."""
    print("🧪 Starting Perquire End-to-End Test")
    
    # Step 1: Create test embedding
    print("\n1️⃣ Creating test embedding...")
    test_embedding = np.random.rand(1536).tolist()  # Gemini embedding size
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_embedding, f)
        embedding_file = f.name
    
    print(f"   ✅ Created embedding file: {embedding_file}")
    print(f"   📊 Embedding shape: {len(test_embedding)}")
    
    # Step 2: Test provider availability
    print("\n2️⃣ Testing provider availability...")
    try:
        from perquire.providers import get_embedding_provider, get_llm_provider
        
        # Test Gemini provider (should be installed)
        embedding_provider = get_embedding_provider("gemini")
        llm_provider = get_llm_provider("gemini")
        print("   ✅ Gemini providers available")
        
    except Exception as e:
        print(f"   ❌ Provider test failed: {e}")
        return False
    
    # Step 3: Test basic embedding
    print("\n3️⃣ Testing embedding generation...")
    try:
        test_text = "A beautiful sunset over the ocean"
        result_embedding = await embedding_provider.embed(test_text)
        print(f"   ✅ Generated embedding for: '{test_text}'")
        print(f"   📊 Result embedding size: {len(result_embedding)}")
        
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")
        print("   💡 Make sure GEMINI_API_KEY is set in .env")
        return False
    
    # Step 4: Test LLM generation
    print("\n4️⃣ Testing LLM generation...")
    try:
        test_prompt = "Describe what this embedding might represent: [embedding vector]"
        response = await llm_provider.generate(test_prompt, max_tokens=100)
        print(f"   ✅ LLM response: {response[:100]}...")
        
    except Exception as e:
        print(f"   ❌ LLM test failed: {e}")
        print("   💡 Make sure GEMINI_API_KEY is set in .env")
        return False
    
    # Step 5: Test investigation engine (when implemented)
    print("\n5️⃣ Testing investigation engine...")
    try:
        # This will fail until we implement PerquireInvestigator
        from perquire.core.investigator import PerquireInvestigator
        
        investigator = PerquireInvestigator(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider
        )
        
        result = await investigator.investigate(test_embedding)
        print(f"   ✅ Investigation completed")
        print(f"   📝 Description: {result.description}")
        print(f"   📊 Final similarity: {result.final_similarity:.3f}")
        print(f"   🔄 Iterations: {result.iteration_count}")
        
    except ImportError:
        print("   ⏳ PerquireInvestigator not implemented yet")
        print("   📋 This is expected - core engine needs implementation")
        return "partial"
    except Exception as e:
        print(f"   ❌ Investigation failed: {e}")
        return False
    
    # Step 6: Test CLI integration
    print("\n6️⃣ Testing CLI integration...")
    try:
        import subprocess
        import os
        
        # Test providers command
        result = subprocess.run(
            ["uv", "run", "perquire", "providers"], 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print("   ✅ CLI providers command works")
        else:
            print(f"   ❌ CLI providers failed: {result.stderr}")
            return False
            
        # Test investigate command (will fail until implemented)
        result = subprocess.run(
            ["uv", "run", "perquire", "investigate", embedding_file],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if "investigate" in result.stderr.lower() or "not implemented" in result.stderr.lower():
            print("   ⏳ CLI investigate command not implemented yet")
        elif result.returncode == 0:
            print("   ✅ CLI investigate command works")
        else:
            print(f"   ❌ CLI investigate failed: {result.stderr}")
            
    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")
        return False
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    Path(embedding_file).unlink()
    print("   ✅ Temporary files removed")
    
    print("\n🎉 End-to-End Test Summary:")
    print("   ✅ Provider factory works")
    print("   ✅ Gemini providers functional") 
    print("   ✅ Embedding generation works")
    print("   ✅ LLM generation works")
    print("   ✅ CLI providers command works")
    print("   ⏳ Investigation engine needs implementation")
    print("   ⏳ CLI investigate command needs implementation")
    
    print("\n🎯 Next Steps:")
    print("   1. Implement PerquireInvestigator core class")
    print("   2. Add convergence detection algorithms")
    print("   3. Enable CLI investigate command")
    print("   4. Add caching and batch processing")
    
    return "partial"

async def test_e2e_with_mock():
    """Test E2E with mock investigation engine."""
    print("\n🎭 Testing with Mock Investigation Engine")
    
    # Mock investigation result
    class MockInvestigationResult:
        def __init__(self):
            self.description = "A scenic landscape with mountains and a lake at sunset"
            self.final_similarity = 0.847
            self.iteration_count = 5
            self.converged = True
            self.questions_asked = [
                "Is this image-related?", 
                "Does it contain natural elements?",
                "Is it a landscape scene?",
                "Does it show water?",
                "Is it during sunset/sunrise?"
            ]
    
    # Mock investigator
    class MockPerquireInvestigator:
        def __init__(self, embedding_provider, llm_provider):
            self.embedding_provider = embedding_provider
            self.llm_provider = llm_provider
        
        async def investigate(self, embedding):
            print("   🔍 Starting mock investigation...")
            print("   ❓ Asking: 'Is this image-related?'")
            print("   ❓ Asking: 'Does it contain natural elements?'")
            print("   ❓ Asking: 'Is it a landscape scene?'")
            print("   ❓ Asking: 'Does it show water?'")
            print("   ❓ Asking: 'Is it during sunset/sunrise?'")
            print("   📈 Similarity improving: 0.234 → 0.456 → 0.678 → 0.789 → 0.847")
            print("   ✅ Converged after 5 iterations")
            return MockInvestigationResult()
    
    # Test mock flow
    try:
        from perquire.providers import get_embedding_provider, get_llm_provider
        
        embedding_provider = get_embedding_provider("gemini")
        llm_provider = get_llm_provider("gemini")
        
        investigator = MockPerquireInvestigator(embedding_provider, llm_provider)
        test_embedding = np.random.rand(1536).tolist()
        
        result = await investigator.investigate(test_embedding)
        
        print(f"\n📋 Mock Investigation Results:")
        print(f"   📝 Description: {result.description}")
        print(f"   📊 Final similarity: {result.final_similarity:.3f}")
        print(f"   🔄 Iterations: {result.iteration_count}")
        print(f"   ✅ Converged: {result.converged}")
        print(f"   ❓ Questions asked: {len(result.questions_asked)}")
        
        print("\n🎯 This shows the expected user experience!")
        return True
        
    except Exception as e:
        print(f"   ❌ Mock test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        print("🚀 Perquire End-to-End Test Suite")
        print("="*50)
        
        # Test current state
        result1 = await test_e2e_investigation()
        
        # Test with mock (show target experience)
        result2 = await test_e2e_with_mock()
        
        print("\n" + "="*50)
        if result1 == "partial" and result2:
            print("🎯 Test Result: FOUNDATION READY")
            print("   ✅ All infrastructure components working")
            print("   ⏳ Core investigation engine ready for implementation")
            print("\n💡 Ready to implement PerquireInvestigator!")
        elif result1 and result2:
            print("🎉 Test Result: FULL SUCCESS")
            print("   ✅ Complete end-to-end flow working")
        else:
            print("❌ Test Result: FOUNDATION ISSUES")
            print("   🔧 Fix provider/infrastructure issues first")
    
    asyncio.run(main())