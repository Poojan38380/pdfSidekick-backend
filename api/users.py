# from fastapi import APIRouter, Depends, HTTPException, status
# from typing import Dict, Any

# from database import get_connection, get_user_by_id
# from schemas import UserResponse
# from utils.auth import get_current_user
# from utils.colorLogger import print_info, print_error

# router = APIRouter()

# # @router.get("/me", response_model=UserResponse)
# async def get_current_user_profile(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
#     """
#     Get the current user's profile
#     """
#     try:
#         return {
#             "id": current_user["id"],
#             "username": current_user["username"],
#             "first_name": current_user["first_name"],
#             "last_name": current_user["last_name"],
#             "email": current_user["email"],
#             "profile_pic": current_user["profile_pic"],
#             "created_at": current_user["created_at"],
#             "updated_at": current_user["updated_at"]
#         }
#     except Exception as e:
#         print_error(f"Error retrieving user profile: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving user profile: {str(e)}"
#         )

# @router.get("/{user_id}", response_model=UserResponse)
# async def get_user_profile(user_id: str) -> Dict[str, Any]:
#     """
#     Get a user profile by ID
#     """
#     pool = await get_connection()
    
#     try:
#         user = await get_user_by_id(pool, user_id)
#         if not user:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="User not found"
#             )
        
#         # Don't return sensitive information
#         return {
#             "id": user["id"],
#             "username": user["username"],
#             "first_name": user["first_name"],
#             "last_name": user["last_name"],
#             "email": user["email"],
#             "profile_pic": user["profile_pic"],
#             "created_at": user["created_at"],
#             "updated_at": user["updated_at"]
#         }
#     except Exception as e:
#         print_error(f"Error retrieving user profile: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving user profile: {str(e)}"
#         ) 